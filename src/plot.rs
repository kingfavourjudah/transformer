use plotters::prelude::*;

pub fn plot_loss(losses: &Vec<f64>) {
    let root = BitMapBackend::new("loss_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss per Epoch", ("sans-serif", 40).into_font())
        .margin(3)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..losses.len(), min_loss..max_loss)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new((0..).zip(losses.iter().cloned()), &RED))
        .unwrap();

    root.present().unwrap();
}
