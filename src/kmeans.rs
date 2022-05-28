use image::{Rgb, Pixel};
use rayon::prelude::*;

pub trait KmeansPoint: Clone + Sized + Send + Sync {
    fn get_squared_distance(&self, point: &Self) -> f32;

    fn from_mean(points: &Vec<&Self>) -> Self;

    fn get_color_format(&self) -> String;

    fn get_color_format_with_palette(&self) -> String;

    fn get_nearest_cluster_distance(&self, from: &Vec<Cluster<Self>>) -> f32 {
        from.iter().fold(f32::INFINITY, |a, b| a.min(self.get_squared_distance(&b.point)))
    }

    fn get_nearest_cluster_index(&self, from: &Vec<Cluster<Self>>) -> u8 {
        let mut nearest: u8 = 0;
        let mut dist: f32;
        let mut min: f32 = f32::INFINITY;

        for p in from {
            dist = p.point.get_squared_distance(self);
            if dist < min {
                min = dist;
                nearest = p.index;
            }
        }
        nearest
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
    pub dataset: Vec<T>,
    pub clusters: Vec<Cluster<T>>,
    pub labels: Vec<u8>
}

impl<T: KmeansPoint> Kmeans<T> {
    pub fn init(dataset: Vec<T>, n_clusters: u8, init_clusters: fn(&mut Self)) -> Self {
        let len = dataset.len();
        let mut _self = Kmeans {
            dataset: dataset,
            clusters: Vec::with_capacity(n_clusters as usize),
            labels: vec![0; len]
        };
        init_clusters(&mut _self);
        _self
    }

    pub fn fit(&mut self) {
        let mut prev_weights: Vec<f32> = vec![1.0; self.clusters.len()];
        let mut feature: f32 = 1.0;
        let mut prev_feature: f32 = 0.0;

        for _ in 0..20 {
            feature = 0.0;
            
            self.assign_points();   
            self.update_clusters();
            
            for i in 0..self.clusters.len() {
                feature += self.clusters[i].weight / prev_weights[i];
                prev_weights[i] = self.clusters[i].weight
            }
            if (feature - prev_feature).abs() < 0.1 {
                break
            }
            prev_feature = feature;
        }
    }

    fn assign_points(&mut self) {
        self.labels = self.dataset
            .par_iter()
            .map(|p| p.get_nearest_cluster_index(&self.clusters))
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

impl KmeansPoint for Rgb<f32> {
    fn get_squared_distance(&self, point: &Self) -> f32 {
        self.channels()
            .iter()
            .zip(point.channels())
            .fold(0.0f32, |acc, (&p1, &p2)| acc + (p2-p1)*(p2-p1))
    }

    fn from_mean(points: &Vec<&Self>) -> Self {
        let (mut r, mut g, mut b): (f32, f32, f32) = (0.0, 0.0, 0.0);
        for p in points {
            r = r + p[0];
            g = g + p[1];
            b = b + p[2];
        }
        let len = points.len() as f32;
        Self::from([r/len, g/len, b/len])
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