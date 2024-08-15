//! https://debuggercafe.com/opencv-hog-for-accurate-and-fast-person-detection/

use std::env;

use objdetect::HOGDescriptor;
use opencv::core::{CommandLineParser, Size, StsBadArg, Vector, Rect};
use opencv::prelude::*;
use opencv::{highgui, imgcodecs, imgproc, objdetect, Error, Result};

fn main() -> Result<()> {
	let args = env::args().collect::<Vec<_>>();
	let args = args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
	let parser = CommandLineParser::new(
		i32::try_from(args.len()).expect("Too many arguments"),
		&args,
		concat!(
			"{help  h           |            | Print this message. }",
			"{image i           |            | Path to the input image. }",
            "{threshold th      | 0.13       | Threshold for accuracy. }", // high (0.7), moderate (0.3), low (0.13)
			"{save s            | false      | Set true to save results.}"),
	)?;

	if parser.has("help")? {
		parser.print_message()?;
		return Ok(());
	}

	let save = parser.get_bool_def("save")?;
    let threshold = parser.get_f64_def("threshold")?;

    let mut hog_descriptor = HOGDescriptor::default()?;
    let people_detector = HOGDescriptor::get_default_people_detector()?;
    let _ = hog_descriptor.set_svm_detector(&people_detector);

	// If input is an image
	if parser.has("image")? {
		let input = parser.get_str_def("image")?;
		let image = imgcodecs::imread_def(&input)?;
		if image.empty() {
			eprintln!("Cannot read image: {}", input);
			return Err(Error::new(StsBadArg, "Cannot read image"));
		}

		let mut image_out = Mat::default();
		imgproc::resize(
			&image,
			&mut image_out,
            image.size()?,
			0.,
			0.,
			imgproc::INTER_LINEAR,
		)?;

        let mut image_gray = Mat::default();
        imgproc::cvt_color_def(&image, &mut image_gray, imgproc::COLOR_BGR2GRAY)?;

        let mut locations = Vector::new();
        let mut weights = Vector::new();

        let hit_threshold = 0.0;
        let win_stride = Size::new(2, 2);
        let padding = Size::new(10, 10);
        let scale = 1.02;
        let group_threshold = 1.0;
        let use_meanshift_grouping = false;

        hog_descriptor.detect_multi_scale_weights(
            &image_gray,
            &mut locations,
            &mut weights,
            hit_threshold,
            win_stride,
            padding,
            scale,
            group_threshold,
            use_meanshift_grouping)?;

        for (location, weight) in locations.into_iter().zip(weights.into_iter()) {
            let people = Rect::new(location.x, location.y, location.width, location.height);
            if weight > threshold {
                imgproc::rectangle_def(&mut image_out, people, (0, 255, 0).into())?;
            }
        }
        
		// Save results if save is true
		if save {
			println!("Saving assets/result.jpg...");
			imgcodecs::imwrite_def("assets/result.jpg", &image_out)?;
		}

		// Visualize results
		highgui::imshow("image", &image_out)?;
		highgui::poll_key()?;

		println!("Press any key to exit...");
		highgui::wait_key(0)?;
	}
    
	println!("Done.");
	Ok(())
}
