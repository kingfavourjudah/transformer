#[cfg(test)]
mod tests {
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_data_generation_shapes() {
        let vocab: Vec<char> = "abc".chars().collect();
        let (inputs, targets) = generate_synthetic_data(&vocab, 4, 10);
        assert_eq!(inputs.len(), 10);
        assert_eq!(targets.len(), 10);
        for (input, target) in inputs.iter().zip(targets.iter()) {
            assert_eq!(input.size(), [3]);
            assert_eq!(target.size(), [3]);
        }
    }
}
