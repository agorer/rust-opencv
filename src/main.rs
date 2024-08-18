use std::env;

use opencv::prelude::*;
use opencv::{Result, imgcodecs, highgui};

mod commandline;
mod yolo;

fn visualize_results(img: Mat) -> Result<()> {
    highgui::imshow("image", &img)?;
	highgui::poll_key()?;
    
	println!("Press any key to exit...");
	highgui::wait_key(0)?;

    Ok(())
}

fn main() -> Result<()> {
    let args = commandline::parse(env::args().collect::<Vec<_>>())?;
    let model_config = yolo::ModelConfig {
        model_path: args.model_path,
        class_names: vec!["person".to_string()],
        input_size: 640,
    };

    // load the model
    let mut model = match yolo::load_model(model_config) {
        Ok(model) => model,
        Err(e) => {
            println!("Error: {}", e);
            std::process::exit(0);
        }
    };
    
    // read the input test image
    let mut img = imgcodecs::imread(&args.image, imgcodecs::IMREAD_COLOR)?;
    if img.size()?.width > 0 {
        println!("Image loaded successfully.");
    } else {
        println!("Failed to load image.");
        std::process::exit(0);
    }

    // change the threshold if needed
    // first threshold is for class confidence score, the second threshold is for NMS boxes
    let detections = yolo::detect(&mut model, &img, 0.5, 0.5); 
    if detections.is_err() {
        println!("Failed to detect, {:?}", detections.err().unwrap());
        std::process::exit(0);
    }

    let detections = detections.unwrap();
    println!("{:?}", detections); 
    yolo::draw_predictions(&mut img, &detections, &model.model_config);

    if args.save {
        println!("Saving assets/result.jpg...");
 		imgcodecs::imwrite_def("assets/result.jpg", &img)?;
    }

    visualize_results(img)?;
    
	println!("Done.");
	Ok(())
}
