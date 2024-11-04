use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    type InferenceSession;

    #[wasm_bindgen(constructor)]
    fn new() -> InferenceSession;

    #[wasm_bindgen(method, catch)]
    async fn create(this: &InferenceSession, model_path: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(method, catch)]
    async fn run(this: &InferenceSession, feeds: JsValue) -> Result<Jsvalue, JsValue>;
}

#[derive(Serialize, Deserialize)]
struct ModelOptions {
    execution_providers: Vec<String>,
    graph_optimization_level: String,
}

#[wasm_bindgen]
pub struct Model {
    session: InferenceSession,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub async fn new(model_path: &str) -> Result<Model, JsValue> {
        let options = ModelOptions {
            execution_providers: vec!["wasm".to_string(), "webgl".to_string()],
            graph_optimization_level: "all".to_string(),
        };

        let session = InferenceSession::new();
        session.created(model_path).await?;

        // get input input_metadata
        let input_names = session.input_names();
        let input_name = input_names
            .first()
            .ok_or_else(|| JsValue::from_str("invalid input shape"))?;

        let input_type = session.getInputType(input_name);
        let input_shape: Vec<usize> = js_sys::Reflect::get(&input_type, &"shape".into())?
            .as_f64()
            .map(|n| n as usize)
            .ok_or_else(|| JsValue::from_str("invalid input shape"))?;

        let input_metadata = TensorMetadata {
            name: input_name.clone(),
            shape: input_shape,
            data_type: "float32".to_string(),
        };

        let output_names = session.outputNames();
        let output_name = output_names
            .first()
            .ok_or_else(|| JsValue::from_str("no output found in model"))?;
        let output_type = session.getOutputType(output_name);
        let output_shape: Vec<usize> = js_sys::Reflect::get(&output_type, &"shape".into())?
            .as_f64()
            .map(|n| n as usize)
            .ok_or_else(|| JsValue::from_str("Invalid output shape"))?;

        let output_metadata = TensorMetadata {
            name: output_name.clone(),
            shape: output_shape,
            data_type: "float32".to_string(),
        };

        Ok(Model {
            session,
            input_shape,
            output_shape,
        })
    }

    #[wasm_bindgen]
    pub async fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>, JsValue> {
        // // Validate input shape
        if input.len() != self.input_metadata.shape.iter().product() {
            return Err(JsValue::from_str(
                "Input size does not match model requirements",
            ));
        }
        // prepare input tensor
        let feeds = js_sys::Object::new();
        let input_tensor = self.create_tensor("float32", &input, &self.input_shape)?;
        js_sys::Reflect::set(&feeds, &"input".into(), &input_tensor)?;

        // Run inference
        let outputs = self.session.run(feeds).await?;

        // Extract results
        let output_data = js_sys::Reflect::get(&outputs, &"output".into())?;
        let data = js_sys::Reflect::get(&output_data, &"data".into())?;

        // Convert to Vec<f32>
        let array = js_sys::Float32Array::new(&data);
        let mut result = vec![0f32; array.length() as usize];
        array.copy_to(&mut result);

        Ok(result)
    }

    #[wasm_bindgen]
    pub fn get_output_shape(&self) -> Vec<usize> {
        self.input_metadata.shape.clone()
    }

    #[wasm_bindgen]
    pub fn get_output_shape(&self) -> Vec<usize> {
        self.output_metadata.shape.clone()
    }

    fn create_tensor(
        &self,
        data_type: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, "dims".into(), &serde_wasm_bindgen::to_value(shape)?)?;
        let array = js_sys::Float32Array::new_with_length(data.len() as u32);
        array.copy_from(data);
        js_sys::Reflect::set(&obj, &"data".into(), &array)?;
        js_sys::Reflect::set(&obj, &"type".into(), &data_type.into())?;
        Ok(obj)
    }

    #[wasm_bindgen]
    pub fn check_gpu_available() -> bool {
        // Check if WebGL2 is available as a proxy for GPU capability
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document
            .create_element("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        web_sys::WebGl2RenderingContext::new(&canvas).is_ok()
    }
}
