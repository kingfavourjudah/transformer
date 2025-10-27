use crate::model::AdvancedGPT;
use tch::{Kind, Tensor};

pub fn evaluate(model: &AdvancedGPT, inputs: &[Tensor], targets: &[Tensor]) -> f64 {
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    for (input, target) in inputs.iter().zip(targets.iter()) {
        let logits = model.forward(input);
        let predicted = logits.argmax(-1, false);
        let correct = predicted
            .eq_tensor(target)
            .to_kind(Kind::Int64)
            .sum(Kind::Int64);

        correct_predictions += correct.int64_value(&[]);
        total_predictions += target.size()[0];
    }

    correct_predictions as f64 / total_predictions as f64
}
