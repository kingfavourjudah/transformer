use tch::{
    Tensor,
    nn::{self, Module, Path},
};

#[derive(Debug)]
pub struct AdvancedGPT {
    token_embedding: nn::Embedding,
    linear_q: nn::Linear,
    linear_k: nn::Linear,
    linear_v: nn::Linear,
    linear_out: nn::Linear,
    feed_forward: nn::Sequential,
    layer_norm_1: nn::LayerNorm,
    layer_norm_2: nn::LayerNorm,
    output_linear: nn::Linear,
}

impl AdvancedGPT {
    pub fn new(vs: &Path, vocab_size: i64, hidden_dim: i64) -> Self {
        let token_embedding = nn::embedding(
            vs / "token_embedding",
            vocab_size,
            hidden_dim,
            Default::default(),
        );
        let linear_q = nn::linear(vs / "linear_q", hidden_dim, hidden_dim, Default::default());
        let linear_k = nn::linear(vs / "linear_k", hidden_dim, hidden_dim, Default::default());
        let linear_v = nn::linear(vs / "linear_v", hidden_dim, hidden_dim, Default::default());
        let linear_out = nn::linear(
            vs / "linear_out",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );

        let feed_forward = nn::seq()
            .add(nn::linear(
                vs / "ffn1",
                hidden_dim,
                hidden_dim * 4,
                Default::default(),
            ))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(
                vs / "ffn2",
                hidden_dim * 4,
                hidden_dim,
                Default::default(),
            ));

        let layer_norm_1 =
            nn::layer_norm(vs / "layer_norm_1", vec![hidden_dim], Default::default());
        let layer_norm_2 =
            nn::layer_norm(vs / "layer_norm_2", vec![hidden_dim], Default::default());
        let output_linear = nn::linear(
            vs / "output_linear",
            hidden_dim,
            vocab_size,
            Default::default(),
        );

        Self {
            token_embedding,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            feed_forward,
            layer_norm_1,
            layer_norm_2,
            output_linear,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let embeddings = self.token_embedding.forward(x);
        let q = self.linear_q.forward(&embeddings);
        let k = self.linear_k.forward(&embeddings);
        let v = self.linear_v.forward(&embeddings);
        let attention_output = q
            .matmul(&k.transpose(-2, -1))
            .softmax(-1, tch::Kind::Float)
            .matmul(&v);
        let attention_output = self.linear_out.forward(&attention_output);
        let normed_attention = self.layer_norm_1.forward(&(embeddings + attention_output));
        let ff_output = self.feed_forward.forward(&normed_attention);
        let normed_output = self.layer_norm_2.forward(&(normed_attention + ff_output));
        self.output_linear.forward(&normed_output)
    }
}
