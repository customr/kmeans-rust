mod initialize;
mod kmeans;
mod rgb;

pub use crate::initialize::{init_plus_plus, init_simple};
pub use crate::kmeans::{Cluster, DistanceMetric, DistanceMetrics, Kmeans, KmeansPoint};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialize::init_plus_plus;
    use image::Rgb;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    const SEED: u64 = 523342;
    const N_CLUSTERS: u8 = 5;

    type Point = Rgb<f32>;

    macro_rules! generate_dataset {
        ($count:expr, $rng:expr) => {{
            let normal: Normal<f32> = Normal::new(0.5, 0.5).unwrap();
            let mut dataset: Vec<Point> = Vec::new();
            for _ in 0..$count {
                dataset.push(Point::from([
                    normal.sample($rng).abs() % 1.0,
                    normal.sample($rng).abs() % 1.0,
                    normal.sample($rng).abs() % 1.0,
                ]))
            }
            dataset
        }};
    }

    #[test]
    fn find_squared_distance() {
        let point1 = Point::from([20.0, 30.0, 60.0]);
        let point2 = Point::from([75.0, 30.0, 80.0]);

        assert_eq!(
            point1.get_distance(&point2, &DistanceMetric::Squared),
            3425.0
        )
    }

    #[test]
    fn find_manhattan_distance() {
        let point1 = Point::from([2., 2., 2.]);
        let point2 = Point::from([1., 1., 1.]);

        assert_eq!(
            point1.get_distance(&point2, &DistanceMetric::Manhattan),
            3.0
        )
    }

    #[test]
    fn find_nearest() {
        let points = vec![
            Cluster {
                point: Point::from([255.0, 255.0, 255.0]),
                index: 0,
                weight: 0.0,
            },
            Cluster {
                point: Point::from([10.0, 50.0, 15.0]),
                index: 1,
                weight: 0.0,
            },
        ];
        let point = Point::from([50.0, 60.0, 30.0]);

        assert!(point.get_nearest_cluster_index(&points, &DistanceMetric::Squared) == 1)
    }

    #[test]
    fn test_init_plus_plus() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let dataset = generate_dataset!(1024, &mut rng);
        let _ = Kmeans::init(dataset, N_CLUSTERS, init_plus_plus, DistanceMetric::Squared);
    }
}
