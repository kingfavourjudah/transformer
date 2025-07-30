#[cfg(test)]
mod tests {
    use crate::model::AdvancedGPT;
    use tch::{Device, Tensor, nn};

    #[test]
    fn test_model_output_shape() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let model = AdvancedGPT::new(&vs.root(), 10, 32);

        let input = Tensor::of_slice(&[1, 2, 3]).reshape(&[3]).to_device(device);
        let output = model.forward(&input);

        assert_eq!(output.size()[..2], [3, 10]);
    }
}
