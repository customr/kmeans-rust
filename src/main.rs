use kmeans::{Kmeans, KmeansPoint, init_plus_plus, init_simple};
use std::time::Instant;
use image::{Rgb};


fn main() {
    let mut input = String::new();
    let mut n_clusters = String::new();
    let mut init_type = String::new();
    println!("Введите путь до изображения: ");
    std::io::stdin().read_line(&mut input).unwrap();
    println!("Введите количество кластеров: ");
    std::io::stdin().read_line(&mut n_clusters).unwrap();
    println!("Выберите метод: \n1 - K-средних\n2 - K-средних++");
    std::io::stdin().read_line(&mut init_type).unwrap();

    let dataset = image::open(input.trim())
        .unwrap()
        .into_rgb32f()
        .pixels()
        .to_owned()
        .collect::<Vec<&Rgb<f32>>>()
        .iter()
        .map(|&p| *p)
        .collect::<Vec<Rgb<f32>>>();
    
    let init_clusters: fn(&mut Kmeans<Rgb<f32>>);
    match init_type.trim() {
        "1" => init_clusters = init_simple,
        _ => init_clusters = init_plus_plus
    }

    let start = Instant::now();
    let mut kmeans = Kmeans::init(
        dataset, 
        n_clusters.trim().parse::<u8>().unwrap(),
        init_clusters
    );
    let duration = start.elapsed();
    println!("Затраченное время на инициализацию: {:?}", duration);

    let start = Instant::now();
    kmeans.fit();
    let duration = start.elapsed();
    println!("Затраченное время на кластеризацию: {:?}", duration);

    println!("Полученная палитра:");
    for c in kmeans.clusters {
        println!("{} {:.2}", c.point.get_color_format_with_palette(), c.weight)
    }
}