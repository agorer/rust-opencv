//! Port of https://github.com/opencv/opencv/blob/4.9.0/samples/dnn/face_detect.cpp
//! Tutorial: https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html

use std::env;

use objdetect::FaceDetectorYN;
use opencv::core::{CommandLineParser, Point, Point2f, Rect2f, Size, StsBadArg, StsError, TickMeter};
use opencv::prelude::*;
use opencv::{highgui, imgcodecs, imgproc, objdetect, Error, Result};

fn visualize(input: &mut Mat, frame: i32, faces: &Mat, fps: f64, thickness: i32) -> Result<()> {
	let fps_string = format!("FPS : {:.2}", fps);
	if frame >= 0 {
		println!("Frame {}, ", frame);
	}
	println!("FPS: {}", fps_string);
	for i in 0..faces.rows() {
		// Print results
		println!(
			"Face {i}, top-left coordinates: ({}, {}), box width: {}, box height: {}, score: {:.2}",
			faces.at_2d::<f32>(i, 0)?,
			faces.at_2d::<f32>(i, 1)?,
			faces.at_2d::<f32>(i, 2)?,
			faces.at_2d::<f32>(i, 3)?,
			faces.at_2d::<f32>(i, 14)?
		);

		// Draw bounding box
		let rect = Rect2f::new(
			*faces.at_2d::<f32>(i, 0)?,
			*faces.at_2d::<f32>(i, 1)?,
			*faces.at_2d::<f32>(i, 2)?,
			*faces.at_2d::<f32>(i, 3)?,
		)
		.to::<i32>()
		.ok_or_else(|| Error::new(StsBadArg, "Invalid rect"))?;
		imgproc::rectangle(input, rect, (0., 255., 0.).into(), thickness, imgproc::LINE_8, 0)?;
		// Draw landmarks
		imgproc::circle(
			input,
			Point2f::new(*faces.at_2d::<f32>(i, 4)?, *faces.at_2d::<f32>(i, 5)?)
				.to::<i32>()
				.ok_or_else(|| Error::new(StsBadArg, "Invalid point"))?,
			2,
			(255., 0., 0.).into(),
			thickness,
			imgproc::LINE_8,
			0,
		)?;
		imgproc::circle(
			input,
			Point2f::new(*faces.at_2d::<f32>(i, 6)?, *faces.at_2d::<f32>(i, 7)?)
				.to::<i32>()
				.ok_or_else(|| Error::new(StsBadArg, "Invalid point"))?,
			2,
			(0., 0., 255.).into(),
			thickness,
			imgproc::LINE_8,
			0,
		)?;
		imgproc::circle(
			input,
			Point2f::new(*faces.at_2d::<f32>(i, 8)?, *faces.at_2d::<f32>(i, 9)?)
				.to::<i32>()
				.ok_or_else(|| Error::new(StsBadArg, "Invalid point"))?,
			2,
			(0., 255., 0.).into(),
			thickness,
			imgproc::LINE_8,
			0,
		)?;
		imgproc::circle(
			input,
			Point2f::new(*faces.at_2d::<f32>(i, 10)?, *faces.at_2d::<f32>(i, 11)?)
				.to::<i32>()
				.ok_or_else(|| Error::new(StsBadArg, "Invalid point"))?,
			2,
			(255., 0., 255.).into(),
			thickness,
			imgproc::LINE_8,
			0,
		)?;
		imgproc::circle(
			input,
			Point2f::new(*faces.at_2d::<f32>(i, 12)?, *faces.at_2d::<f32>(i, 13)?)
				.to::<i32>()
				.ok_or_else(|| Error::new(StsBadArg, "Invalid point"))?,
			2,
			(0., 255., 255.).into(),
			thickness,
			imgproc::LINE_8,
			0,
		)?;
	}
	imgproc::put_text(
		input,
		&fps_string,
		Point::new(0, 15),
		imgproc::FONT_HERSHEY_SIMPLEX,
		0.5,
		(0., 255., 0.).into(),
		thickness,
		imgproc::LINE_8,
		false,
	)?;
	Ok(())
}

fn main() -> Result<()> {
	let args = env::args().collect::<Vec<_>>();
	let args = args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
	let parser = CommandLineParser::new(
		i32::try_from(args.len()).expect("Too many arguments"),
		&args,
		concat!(
			"{help  h           |            | Print this message}",
			"{image i           |            | Path to the input image1. Omit for detecting through VideoCapture}",
            "{scale sc          | 1.0        | Scale factor used to resize input video frames}",
			"{fd_model fd       | models/face_detection_yunet_2023mar.onnx | Path to the model. Download yunet.onnx in https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet}",
			"{score_threshold   | 0.9        | Filter out faces of score < score_threshold}",
			"{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold}",
			"{top_k             | 5000       | Keep top_k bounding boxes before NMS}",
			"{save s            | false      | Set true to save results. This flag is invalid when using camera}"),
	)?;

	if parser.has("help")? {
		parser.print_message()?;
		return Ok(());
	}

	let fd_model_path = parser.get_str_def("fd_model")?;

	let score_threshold = parser.get_f64_def("score_threshold")? as f32;
	let nms_threshold = parser.get_f64_def("nms_threshold")? as f32;
	let top_k = parser.get_i32_def("top_k")?;

	let save = parser.get_bool_def("save")?;
	let scale = parser.get_f64_def("scale")?;

	// Initialize FaceDetectorYN
	let mut detector = FaceDetectorYN::create(
		&fd_model_path,
		"",
		Size::new(320, 320),
		score_threshold,
		nms_threshold,
		top_k,
		0,
		0,
	)?;

	let mut tm = TickMeter::default()?;

	// If input is an image
	if parser.has("image")? {
		let input = parser.get_str_def("image")?;
		let image = imgcodecs::imread_def(&input)?;
		if image.empty() {
			eprintln!("Cannot read image: {}", input);
			return Err(Error::new(StsBadArg, "Cannot read image"));
		}

		let image_width = (f64::from(image.cols()) * scale) as i32;
		let image_height = (f64::from(image.rows()) * scale) as i32;
		let mut image_out = Mat::default();
		imgproc::resize(
			&image,
			&mut image_out,
			Size::new(image_width, image_height),
			0.,
			0.,
			imgproc::INTER_LINEAR,
		)?;
		let mut image = image_out;

		tm.start()?;

		// Set input size before inference
		detector.set_input_size(image.size()?)?;

		let mut faces = Mat::default();
		detector.detect(&image, &mut faces)?;
		if faces.rows() < 1 {
			eprintln!("Cannot find a face in {input}");
			return Err(Error::new(StsError, "Cannot find a face"));
		}

		tm.stop()?;

		// Draw results on the input image
		visualize(&mut image, -1, &faces, tm.get_fps()?, 2)?;

		// Save results if save is true
		if save {
			println!("Saving result.jpg...");
			imgcodecs::imwrite_def("result.jpg", &image)?;
		}

		// Visualize results
		highgui::imshow("image", &image)?;
		highgui::poll_key()?;

		println!("Press any key to exit...");
		highgui::wait_key(0)?;
	}
    
	println!("Done.");
	Ok(())
}
