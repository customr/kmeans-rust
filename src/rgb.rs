use crate::{DistanceMetric, DistanceMetrics, KmeansPoint};
use image::{Pixel, Rgb};

impl DistanceMetrics for Rgb<f32> {
    fn get_squared_distance(&self, point: &Self) -> f32 {
        self.channels()
            .iter()
            .zip(point.channels())
            .fold(0.0f32, |acc, (&p1, &p2)| acc + (p2 - p1) * (p2 - p1))
    }

    fn get_manhattan_distance(&self, point: &Self) -> f32 {
        self.channels()
            .iter()
            .zip(point.channels())
            .fold(0.0f32, |acc, (&p1, &p2)| acc + (p2 - p1).abs())
    }
}

impl KmeansPoint for Rgb<f32> {
    fn get_distance(&self, point: &Self, metric: &DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Squared => self.get_squared_distance(point),
            DistanceMetric::Manhattan => self.get_manhattan_distance(point),
        }
    }

    fn from_mean(points: &Vec<&Self>) -> Self {
        let (mut r, mut g, mut b): (f32, f32, f32) = (0.0, 0.0, 0.0);
        for p in points {
            r = r + p[0];
            g = g + p[1];
            b = b + p[2];
        }
        let len = points.len() as f32;
        Self::from([r / len, g / len, b / len])
    }

    fn get_color_format(&self) -> String {
        format!(
            "#{:02x}{:02x}{:02x}",
            (self[0] * 256.0) as u8,
            (self[1] * 256.0) as u8,
            (self[2] * 256.0) as u8
        )
    }

    fn get_color_format_with_palette(&self) -> String {
        format!(
            "\x1B[38;2;{};{};{}m▇ {}\x1B[0m",
            (self[0] * 256.0) as u8,
            (self[1] * 256.0) as u8,
            (self[2] * 256.0) as u8,
            self.get_color_format()
        )
    }

    fn get_rgb(&self) -> [u8; 3] {
        return [
            (self[0] * 256.0) as u8,
            (self[1] * 256.0) as u8,
            (self[2] * 256.0) as u8,
        ];
    }
}
