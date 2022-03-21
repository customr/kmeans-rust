use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use crate::{KmeansPoint, Cluster};

pub fn init_plus_plus<T: KmeansPoint> (
    dataset: &Vec<T>,
    clusters: &mut Vec<Cluster<T>>,
    n_clusters: u8,
    rng: &mut impl Rng,
) {
    clusters.push(
        Cluster {
            index: 0,
            point: dataset.choose(rng).unwrap().clone(),
            weight: 0.0
        }
    );
    
    let mut squared_distance: Vec<f32>;
    let mut rnd: f32;
    let mut sum: f32;

    for i in 1..n_clusters {
        squared_distance = dataset
            .clone()
            .par_iter()
            .map(|p| p.get_nearest_cluster(clusters).unwrap().1)
            .collect();

        sum = squared_distance.par_iter().sum::<f32>();
        rnd = rng.gen_range(0.0..1.0) as f32 * sum; 

        sum = 0.0;
        for pi in 0..squared_distance.len() {
            sum = sum + squared_distance[pi];
            if sum > rnd {
                clusters.push(
                    Cluster {
                        index: i,
                        point: dataset[pi].clone(),
                        weight: 0.0
                    }
                );
                break;
            }
        }
    }
}
