use rand::{SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use rayon::prelude::*;
use crate::initialize::init_plus_plus;

const SEED: u64 = 12345;

pub trait KmeansPoint: Clone + Sized + Send + Sync {
    fn get_squared_distance(&self, point: &Self) -> f32;

    fn from_mean(points: &Vec<&Self>) -> Self;

    fn get_color_format(&self) -> String;

    fn get_color_format_with_palette(&self) -> String;

    fn get_nearest_cluster(&self, from: &Vec<Cluster<Self>>) -> Option<(u8, f32)> {
        if from.len() == 0 {
            return None
        } 

        let mut dist: f32;
        let mut min: f32 = f32::MAX;
        let mut nearest: u8 = 0;

        for p in from {
            dist = p.point.get_squared_distance(self);
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

        init_plus_plus(
            &dataset, 
            &mut clusters, 
            n_clusters, 
            &mut rng
        );

        Kmeans {
            dataset: dataset,
            clusters: clusters,
            labels: labels
        }
    }

    pub fn fit(&mut self, n_iter: u16) {
        for _ in 0..n_iter {
            // let start = Instant::now();
            self.labels = self.assign_points();   
            // let duration = start.elapsed();
            // println!("Assign: {:?}", duration);

            // let start = Instant::now();
            self.update_clusters();
            // let duration = start.elapsed();
            // println!("Update: {:?}", duration);
        }
    }

    fn assign_points(&self) -> Vec<u8> {
        self.dataset
            .par_iter()
            .map(|p| p.get_nearest_cluster(&self.clusters).unwrap().0)
            .collect()
    }

    fn update_clusters(&mut self) {
        for c in &mut self.clusters {
            let points: Vec<&T> = self.dataset
                .par_iter()
                .zip(&self.labels)
                .filter(|(_, &l)| l == c.index)
                .map(|(p, _)| p)
                .collect();

            c.point = T::from_mean(&points);
            c.weight = points.len() as f32 / self.dataset.len() as f32
        }
    }
}
