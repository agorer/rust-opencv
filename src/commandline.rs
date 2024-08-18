use opencv::prelude::*;
use opencv::core::{CommandLineParser, StsBadArg};
use opencv::{Result, Error};

pub struct CommandLineArguments {
    pub image: String,
    pub model_path: String,
    pub save: bool,
}

pub fn parse(args: Vec<String>) -> Result<CommandLineArguments> {
    let args = args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
	let parser = CommandLineParser::new(
		i32::try_from(args.len()).expect("Too many arguments"),
		&args,
		concat!(
            "{model m           |            | ONNX formatted model. }",
			"{image i           |            | Path to the input image. }",
			"{save s            | false      | Set true to save results.}"),
	)?;

    if !(parser.has("image")? && parser.has("model")?) {
        return Err(Error::new(StsBadArg, "Missing arguments"));
    }

    Ok(CommandLineArguments{
        image: parser.get_str_def("image")?,
        model_path: parser.get_str_def("model")?,
        save: parser.get_bool_def("save")?,
    })
}
