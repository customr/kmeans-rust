use image::{Pixel, Rgb};
use kmeans::{init_plus_plus, init_simple, DistanceMetric, Kmeans, KmeansPoint};
use std::time::Instant;
fn main() {
    let n_clusters: u8;
    let init_clusters: fn(&mut Kmeans<Rgb<f32>>);
    let metric_type: DistanceMetric;
    let path: &str;
    let dimensions: (u32, u32);

    let mut buf = String::new();
    println!("Введите путь до изображения: ");
    std::io::stdin().read_line(&mut buf).unwrap();
    path = buf.trim();
    let mut img = image::open(path).unwrap().into_rgb32f();
    dimensions = img.dimensions();
    buf.clear();

    println!("Введите количество кластеров: ");
    std::io::stdin().read_line(&mut buf).unwrap();
    n_clusters = buf.trim().parse::<u8>().unwrap();
    buf.clear();

    println!("Выберите метод: \n1 - K-средних\n2 - K-средних++");
    std::io::stdin().read_line(&mut buf).unwrap();
    match buf.trim() {
        "1" => init_clusters = init_simple,
        _ => init_clusters = init_plus_plus,
    }
    buf.clear();

    println!("Выберите метрику расстояния: \n1 - квадратичное\n2 - манхэттен");
    std::io::stdin().read_line(&mut buf).unwrap();
    match buf.trim() {
        "1" => metric_type = DistanceMetric::Squared,
        _ => metric_type = DistanceMetric::Manhattan,
    }
    buf.clear();
    let dataset = img.pixels_mut().map(|p| *p).collect::<Vec<Rgb<f32>>>();

    println!("Кол-во точек: {}", dimensions.0 * dimensions.1);

    let start = Instant::now();
    let mut kmeans = Kmeans::init(dataset, n_clusters, init_clusters, metric_type);
    let duration = start.elapsed();
    println!("Затраченное время на инициализацию: {:?}", duration);

    let start = Instant::now();
    kmeans.fit();
    let duration = start.elapsed();
    println!("Затраченное время на кластеризацию: {:?}", duration);

    println!("\nПолученная палитра:");
    for c in kmeans.clusters.clone() {
        println!(
            "{} {:.2}",
            c.point.get_color_format_with_palette(),
            c.weight
        )
    }
    let imgbuf = image::ImageBuffer::from_fn(dimensions.0, dimensions.1, |x, y| {
        let index_of_cluster = kmeans.labels[(x + y * dimensions.0) as usize];
        let index = kmeans
            .clusters
            .iter()
            .position(|x| x.index == index_of_cluster)
            .unwrap();
        let point = kmeans.clusters.get(index).unwrap().point.channels();
        Rgb::<u8>::from([
            (point[0] * 255.0) as u8,
            (point[1] * 255.0) as u8,
            (point[2] * 255.0) as u8,
        ])
    });
    imgbuf.save("result.png").unwrap();
}
