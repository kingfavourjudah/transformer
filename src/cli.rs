use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "GPT Char Model")]
#[command(about = "Train a simple GPT model on synthetic data", long_about = None)]
pub struct CliArgs {
    /// Number of training samples
    #[arg(short, long, default_value_t = 1000)]
    pub samples: usize,

    /// Sequence length of the training data
    #[arg(short = 'l', long, default_value_t = 5)]
    pub seq_len: usize,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 500)]
    pub epochs: usize,

    /// Hidden dimension size
    #[arg(short = 'd', long, default_value_t = 128)]
    pub hidden_dim: i64,

    /// Learning rate
    #[arg(short, long, default_value_t = 1e-3)]
    pub lr: f64,
}
