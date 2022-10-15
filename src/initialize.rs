use crate::{Cluster, Kmeans, KmeansPoint};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

pub fn init_simple<T: KmeansPoint>(kmeans: &mut Kmeans<T>) {
    let mut rng = StdRng::seed_from_u64(123);
    for i in 0..kmeans.clusters.capacity() as u8 {
        kmeans.clusters.push(Cluster {
            index: i,
            point: kmeans.dataset.choose(&mut rng).unwrap().clone(),
            weight: 0.0,
        });
    }
}

pub fn init_plus_plus<T: KmeansPoint>(kmeans: &mut Kmeans<T>) {
    let mut rng = StdRng::seed_from_u64(123);
    kmeans.clusters.push(Cluster {
        index: 0,
        point: kmeans.dataset.choose(&mut rng).unwrap().clone(),
        weight: 0.0,
    });

    let mut squared_distance: Vec<f32>;
    let mut rnd: f32;
    let mut sum: f32;

    for i in 1..kmeans.clusters.capacity() as u8 {
        squared_distance = kmeans
            .dataset
            .clone()
            .par_iter()
            .map(|p| p.get_nearest_cluster_distance(&kmeans.clusters, &kmeans.distance_metric))
            .collect();

        sum = squared_distance.par_iter().sum::<f32>();
        rnd = rng.gen_range(0.0..1.0) as f32 * sum;

        sum = 0.0;
        for pi in 0..squared_distance.len() {
            sum = sum + squared_distance[pi];
            if sum > rnd {
                kmeans.clusters.push(Cluster {
                    index: i,
                    point: kmeans.dataset[pi].clone(),
                    weight: 0.0,
                });
                break;
            }
        }
    }
}
