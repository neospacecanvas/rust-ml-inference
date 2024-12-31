use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the ONNX Runtime environment
    let environment = Environment::builder()
        .with_name("dynamic_model_inference")
        .with_log_level(onnxruntime::LoggingLevel::Warning)
        .build()?;

    // Load the ONNX model
    let model_path = "iris_model.onnx";
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_model_from_file(model_path)?;

    // Access input and output metadata as fields
    let input_metadata = &session.inputs;
    let output_metadata = &session.outputs;

    // Print input and output details
    println!("Model Inputs:");
    for input in input_metadata {
        println!(
            "  Name: {}, Shape: {:?}, Type: {:?}",
            input.name, input.dimensions, input.input_type
        );
    }

    println!("Model Outputs:");
    for output in output_metadata {
        println!(
            "  Name: {}, Shape: {:?}, Type: {:?}",
            output.name, output.dimensions, output.output_type
        );
    }

    // Assume the first input tensor and create a dummy input
    let input_tensor_shape = input_metadata[0]
        .dimensions
        .iter()
        .map(|dim| match dim {
            Some(d) => *d as usize,
            None => 1, // Use 1 for dynamic dimensions
        })
        .collect::<Vec<_>>();

    let input_tensor: Array<f32, _> = Array::from_shape_fn(input_tensor_shape, |_| 0.0);

    // Run inference
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;

    // Process and print outputs
    for (i, output) in outputs.iter().enumerate() {
        println!("Output {}: {:?}", i, output);
    }

    Ok(())
}
