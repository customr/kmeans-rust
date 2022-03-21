# kmeans-rust
> K-means algorithm in Rust for clustering in any dimensions

## Usage
```rust
const N_CLUSTERS: u8 = 4;
let dataset: Vec<KmeansPoint> = unimplemented!();
let mut kmeans = Kmeans::init(dataset, N_CLUSTERS);
kmeans.fit(2);
for c in kmeans.clusters {
    println!("{} {:.2}", c.point.get_color_format_with_palette(), c.weight)
}
```