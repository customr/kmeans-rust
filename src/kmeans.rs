use rand::{SeedableRng};
use rand::rngs::StdRng;
use crate::initialize;

const SEED: u64 = 12345;

pub trait KmeansPoint: Clone + Sized {
    fn get_squared_distance(&self, point: &Self) -> f32;

    fn get_mean_point(points: &Vec<&Self>) -> Self;

    fn get_color_format(&self) -> String;

    fn get_color_format_with_palette(&self) -> String;

    fn get_nearest_cluster(&self, from: &Vec<Cluster<Self>>) -> Option<(u8, f32)> {
        if from.len() == 0 {
            return None
        } 

        let mut min: f32 = f32::MAX;
        let mut nearest: u8 = from[0].index;

        for p in from {
            let dist = p.point.get_squared_distance(self);
            if dist < min {
                min = dist;
                nearest = p.index;
            }
        }
        Some((nearest, min))
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Cluster<P: KmeansPoint> {
    pub point: P,
    pub index: u8,
    pub weight: f32
}

#[derive(Debug)]
pub struct Kmeans<T: KmeansPoint> {
    dataset: Vec<T>,
    pub clusters: Vec<Cluster<T>>,
    pub labels: Vec<u8>
}

impl<T: KmeansPoint> Kmeans<T> {
    pub fn init(dataset: Vec<T>, n_clusters: u8) -> Self {
        let mut clusters = Vec::with_capacity(n_clusters as usize);
        let labels = vec![0; dataset.len()];
        let mut rng = StdRng::seed_from_u64(SEED);
        initialize::init_plus_plus(&dataset, &mut clusters, n_clusters, &mut rng);

        Kmeans {
            dataset: dataset,
            clusters: clusters,
            labels: labels
        }
    }

    pub fn fit(&mut self, n_iter: u16) {
        for _ in 0..n_iter {
            self.assign_points();   
            self.update_clusters();
        }
    }

    fn assign_points(&mut self) {
        self.labels = self.dataset
            .clone()
            .iter()
            .map(|p| p.get_nearest_cluster(&self.clusters).unwrap().0)
            .collect()
    }

    fn update_clusters(&mut self) {
        for c in &mut self.clusters {
            let points: Vec<&T> = self.dataset
                .iter()
                .zip(&mut self.labels)
                .into_iter()
                .filter(|(_, l)| **l == c.index)
                .map(|(p, _)| p)
                .collect();

            c.point = T::get_mean_point(&points);
            c.weight = points.len() as f32 / self.dataset.len() as f32
        }
    }
}
