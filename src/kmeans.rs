use rayon::prelude::*;

#[derive(Debug)]
pub enum DistanceMetric {
    Squared,
    Manhattan,
}

pub trait DistanceMetrics: Clone + Sized + Send + Sync {
    fn get_squared_distance(&self, point: &Self) -> f32;
    fn get_manhattan_distance(&self, point: &Self) -> f32;
}

pub trait KmeansPoint: DistanceMetrics {
    fn from_mean(points: &Vec<&Self>) -> Self;

    fn get_color_format(&self) -> String;

    fn get_color_format_with_palette(&self) -> String;

    fn get_rgb(&self) -> [u8; 3];

    fn get_distance(&self, point: &Self, metric: &DistanceMetric) -> f32;

    fn get_nearest_cluster_distance(
        &self,
        from: &Vec<Cluster<Self>>,
        metric: &DistanceMetric,
    ) -> f32 {
        from.iter().fold(f32::INFINITY, |a, b| {
            a.min(self.get_distance(&b.point, metric))
        })
    }

    fn get_nearest_cluster_index(&self, from: &Vec<Cluster<Self>>, metric: &DistanceMetric) -> u8 {
        let mut nearest: u8 = 0;
        let mut dist: f32;
        let mut min: f32 = f32::INFINITY;

        for p in from {
            dist = self.get_distance(&p.point, metric);

            if dist < min {
                min = dist;
                nearest = p.index;
            }
        }
        nearest
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Cluster<T: KmeansPoint> {
    pub point: T,
    pub index: u8,
    pub weight: f32,
}

#[derive(Debug)]
pub struct Kmeans<T: KmeansPoint> {
    pub dataset: Vec<T>,
    pub clusters: Vec<Cluster<T>>,
    pub labels: Vec<u8>,
    pub distance_metric: DistanceMetric,
}

impl<T: KmeansPoint> Kmeans<T> {
    pub fn init(
        dataset: Vec<T>,
        n_clusters: u8,
        init_clusters: fn(&mut Self),
        metric: DistanceMetric,
    ) -> Self {
        let len = dataset.len();
        let mut _self = Kmeans {
            dataset: dataset,
            clusters: Vec::with_capacity(n_clusters as usize),
            labels: vec![0; len],
            distance_metric: metric,
        };
        init_clusters(&mut _self);
        _self
    }

    pub fn fit(&mut self) {
        let mut prev_weights: Vec<f32> = vec![1.0; self.clusters.len()];
        let mut feature: f32;
        let mut prev_feature: f32 = 1.0;

        for _ in 0..10 {
            feature = 0.0;

            self.assign_points();
            self.update_clusters();

            for i in 0..self.clusters.len() {
                feature += self.clusters[i].weight / prev_weights[i];
                prev_weights[i] = self.clusters[i].weight
            }
            if (feature / prev_feature) < 0.1 {
                break;
            }
            prev_feature = feature;
        }
        self.clusters
            .sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    }

    fn assign_points(&mut self) {
        self.labels = self
            .dataset
            .par_iter()
            .map(|p| p.get_nearest_cluster_index(&self.clusters, &self.distance_metric))
            .collect()
    }

    fn update_clusters(&mut self) {
        for c in &mut self.clusters {
            let points: Vec<&T> = self
                .dataset
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
