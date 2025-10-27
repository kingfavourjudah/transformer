use rand::{seq::SliceRandom, thread_rng};
use tch::{Kind, Tensor};

pub fn generate_synthetic_data(
    vocab: &[char],
    seq_len: usize,
    num_samples: usize,
) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut rng = thread_rng();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..num_samples {
        let seq: Vec<i64> = (0..seq_len)
            .map(|_| {
                let c = vocab.choose(&mut rng).unwrap();
                vocab.iter().position(|&v| v == *c).unwrap() as i64
            })
            .collect();

        inputs.push(Tensor::from_slice(&seq[..seq_len - 1]).to_kind(Kind::Int64));
        targets.push(Tensor::from_slice(&seq[1..]).to_kind(Kind::Int64));
    }

    (inputs, targets)
}
