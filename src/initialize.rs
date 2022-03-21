use rand::Rng;
use rand::seq::SliceRandom;
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
        squared_distance = dataset.clone().into_iter().map(|p| p.get_nearest_cluster(clusters).unwrap().1).collect();
        rnd = rng.gen_range(0.0..1.0) as f32 * squared_distance.iter().sum::<f32>();
        sum = 0.0;

        'outer: loop {
            for p in dataset {
                sum = sum + p.get_nearest_cluster(clusters).unwrap().1;
                if sum > rnd {
                    clusters.push(
                        Cluster {
                            index: i,
                            point: p.clone(),
                            weight: 0.0
                        }
                    );
                    break 'outer;
                }
            }
            break 'outer;
        }
    }
}
