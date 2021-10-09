use crate::utils::*;
use rand::seq::SliceRandom;
use num_traits::{Num, Float};


#[derive(Debug, PartialEq)]
pub enum KmeansError{
    ZeroInput
}

#[derive(Clone, Debug)]
pub struct KmeansSettings{
    min_delta: f32,
    max_iterations: u16,
    centroids_num: u8
}

#[derive(Clone, Debug)]
pub struct Kmeans<'a, P, const D: usize> 
where P: 'a + Float + Num + PartialOrd + Clone {
    points: &'a Vec<[P; D]>,
    centroids: Vec<[P; D]>,
    labels: Vec<u8>,
    settings: KmeansSettings
}

impl<'a, P, const D:usize> Kmeans<'a, P, D> 
where P: 'a + Float + Num + PartialOrd + Clone {
    pub fn new(
        points: &'a Vec<[P; D]>,
        settings: KmeansSettings
    ) -> Result<Self, KmeansError> {
        if points.len() < 2 {
            Err(
                KmeansError::ZeroInput
            )
        } else {
            Ok(
                Kmeans {
                    points: points,
                    centroids: Vec::new(),
                    labels: vec![0; points.len()],
                    settings: settings
                }
            )
        }
    }
    fn init(&mut self) {
        for _ in 0..self.settings.centroids_num {
            self.centroids.push(
                self.points.choose(&mut rand::thread_rng()).unwrap().clone()
            )
        }
    }
    fn recenter(&mut self) {
        
    }
    fn sse(p1: &[P; D], p2: &[P; D]) -> P {
        let mut sse = P::zero();
        for d in 0..D {
            sse = sse + (p1[d]-p2[d]).powf(P::from(2).unwrap())
        }
        sse
    }
    fn predict(&self, point: &[P; D]) -> u8 {
        let mut centroid: u8 = 0;
        let mut min_dist: P = P::infinity();
        let mut dist: P;
        for c in 0..self.centroids.len() {
            dist = Kmeans::sse(point, &self.centroids[c]);
            if dist < min_dist {
                min_dist = dist;
                centroid = c as u8;
            }
        }
        centroid
    }
    pub fn fit(&mut self) {
        self.init();
        for _ in 1..=self.settings.max_iterations {
            for p in 0..self.points.len() {
                self.labels[p] = self.predict(&self.points[p])
            }
            self.recenter()
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! generate_points {
        ($t:ty, $n:expr, $($count:expr),*) => {
            {
                let mut points: Vec<[$t; $n]> = Vec::new();
                $(
                    for _ in 0..$count {
                        points.push(rand::random::<[$t; $n]>())
                    }
                )*
                points
            }
        };
    }

    #[test]
    fn fit() {
        let points = generate_points!(f32, 2, 8);
        let settings = KmeansSettings {
            min_delta: 0.1,
            max_iterations: 5,
            centroids_num: 3
        };
        let mut kmeans = Kmeans::new(
            &points,
            settings
        ).unwrap();

        kmeans.fit();

        println!("{:#?}", kmeans);
    }
}