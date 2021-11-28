use clap::Parser;

pub fn get_args() -> CliOpts {
    CliOpts::parse()
}

#[derive(Parser, Debug)]
#[clap(version = clap::crate_version!(), author = "Scott S. <scottschroeder@sent.com>")]
pub struct CliOpts {
    #[clap(short, long, parse(from_occurrences))]
    pub verbose: u8,
    #[clap(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(Parser, Debug)]
pub enum SubCommand {
    DemoImageXY(DemoImageXY),
    DemoIndex(DemoIndex),
    Gaussian(Gaussian),
    Test(Test),
}

#[derive(Parser, Debug)]
pub struct DemoImageXY {
    pub path: String,
}
#[derive(Parser, Debug)]
pub struct DemoIndex {
    #[clap(about = "path to astrometry.net fits file")]
    pub path: String,
}

#[derive(Parser, Debug)]
pub struct Gaussian {
    pub path: String,
    pub x: u32,
    pub y: u32,
}

#[derive(Parser, Debug)]
pub struct Test {}
