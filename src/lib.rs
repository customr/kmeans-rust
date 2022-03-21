mod initialize;
mod kmeans;

pub use crate::kmeans::{Cluster, KmeansPoint, Kmeans};


#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialize::init_plus_plus;
    use std::time::Instant;
    use image::{Rgb, Pixel};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Normal, Distribution};

    const SEED: u64 = 523342;
    const N_CLUSTERS: u8 = 6;

    type Point = Rgb<f32>;

    macro_rules! generate_dataset {
        ($count:expr, $rng:expr) => {
            {
                let normal: Normal<f32> = Normal::new(0.5, 0.5).unwrap();
                let mut dataset: Vec<Point> = Vec::new();
                for _ in 0..$count {
                    dataset.push(
                        Point::from([
                            normal.sample($rng).abs() % 1.0,
                            normal.sample($rng).abs() % 1.0,
                            normal.sample($rng).abs() % 1.0
                        ])
                    )
                }
                dataset
            }
        };
    }

    impl KmeansPoint for Point {
        fn get_squared_distance(&self, point: &Self) -> f32 {
            self.channels()
                .iter()
                .zip(point.channels())
                .map(|(p1, p2)| (p2-p1)*(p2-p1))
                .reduce(|acc, p| acc + p)
                .unwrap()
        }

        fn from_mean(points: &Vec<&Self>) -> Self {
            let (mut r, mut g, mut b): (f32, f32, f32) = (0.0, 0.0, 0.0);
            for p in points {
                r = r + p[0];
                g = g + p[1];
                b = b + p[2];
            }
            let len = points.len() as f32;
            Point::from([r/len, g/len, b/len])
        }

        fn get_color_format(&self) -> String {
            format!(
                "#{:02x}{:02x}{:02x}", 
                (self[0]*256.0) as u8, 
                (self[1]*256.0) as u8, 
                (self[2]*256.0) as u8
            )
        }
        
        fn get_color_format_with_palette(&self) -> String {
            format!(
                "\x1B[38;2;{};{};{}m▇ {}\x1B[0m",
                (self[0]*256.0) as u8, 
                (self[1]*256.0) as u8, 
                (self[2]*256.0) as u8,
                self.get_color_format()
            )
        }
    }

    #[test]
    fn find_squared_distance() {
        let point1 = Point::from([20.0,30.0,60.0]);
        let point2 = Point::from([75.0,30.0,80.0]);
        
        assert_eq!(point1.get_squared_distance(&point2), 3425.0)
    }

    #[test]
    fn find_nearest() {
        let points = vec![
            Cluster{
                point: Point::from([255.0,255.0,255.0]),
                index: 0,
                weight: 0.0
            }, 
            Cluster{
                point: Point::from([10.0,50.0,15.0]),
                index: 1,
                weight: 0.0
            }
        ];
        let point = Point::from([50.0,60.0,30.0]);

        assert!(point.get_nearest_cluster(&points).unwrap().0 == 1)
    }

    #[test]
    fn test_init_plus_plus() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut clusters: Vec<Cluster<Point>> = Vec::with_capacity(N_CLUSTERS as usize);
        let dataset = generate_dataset!(1024, &mut rng);
        init_plus_plus(&dataset, &mut clusters, N_CLUSTERS, &mut rng);
    }

    #[test]
    fn kmeans() {
        let dataset = image::open("samples/traffic.jpg").unwrap()
            .into_rgb32f()
            .pixels()
            .collect::<Vec<&Point>>()
            .iter()
            .map(|&p| *p)
            .collect::<Vec<Point>>();
        
        let start = Instant::now();
        let mut kmeans = Kmeans::init(dataset, N_CLUSTERS);
        let duration = start.elapsed();
        println!("Init time elapsed: {:?}", duration);

        let start = Instant::now();
        kmeans.fit(2);
        let duration = start.elapsed();
        println!("Time elapsed: {:?}", duration);

        for c in kmeans.clusters {
            println!("{} {:.2}", c.point.get_color_format_with_palette(), c.weight)
        }
    }
}