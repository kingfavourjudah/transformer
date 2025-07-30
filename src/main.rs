mod cli;
mod data;
mod evaluate;
mod model;
mod plot;

use crate::cli::CliArgs;
use crate::data::generate_synthetic_data;
use crate::evaluate::evaluate;
use crate::model::AdvancedGPT;
use crate::plot::plot_loss;

use clap::Parser;
use tch::{Device, nn};

use tch::nn::OptimizerConfig;

fn main() {
    let args = CliArgs::parse();

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let vocab: Vec<char> = "abcdefghijklmnopqrstuvwxyz ".chars().collect();
    let vocab_size = vocab.len() as i64;

    let (inputs, targets) = generate_synthetic_data(&vocab, args.seq_len, args.samples);

    let model = AdvancedGPT::new(&vs.root(), vocab_size, args.hidden_dim);
    let mut opt = nn::Adam::default().build(&vs, args.lr).unwrap();

    let mut losses = Vec::new();

    for epoch in 1..=args.epochs {
        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let logits = model.forward(input);
            let loss = logits.cross_entropy_for_logits(target);
            opt.backward_step(&loss);
            total_loss += f64::from(loss);
        }

        let avg_loss = total_loss / args.samples as f64;
        losses.push(avg_loss);
        println!("Epoch: {}, Loss: {:.4}", epoch, avg_loss);
    }

    plot_loss(&losses);

    let accuracy = evaluate(&model, &inputs, &targets);
    println!("Evaluation accuracy: {:.4}", accuracy);
}
