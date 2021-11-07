use argparse::CliOpts;

mod argparse;

fn main() -> anyhow::Result<()> {
    color_backtrace::install();
    let args = argparse::get_args();
    setup_logger(args.verbose);
    // cli::setup_logger(args.occurrences_of("verbosity"));
    log::trace!("Args: {:?}", args);

    run(&args).map_err(|e| {
        log::error!("{}", e);
        e.chain()
            .skip(1)
            .for_each(|cause| log::error!("because: {}", cause));
        anyhow::anyhow!("unrecoverable astrometry failure")
    })
}

fn run(args: &CliOpts) -> anyhow::Result<()> {
    match &args.subcmd {
        argparse::SubCommand::DemoImageXY(opts) => imagexy::demo(opts.path.as_str()),
        argparse::SubCommand::Gaussian(opts) => {
            imagexy::gaussian(opts.path.as_str(), opts.x, opts.y)
        }
        argparse::SubCommand::Test(_opts) => scratch(),
    }
}

fn scratch() -> anyhow::Result<()> {
    Ok(())
}

pub fn setup_logger(level: u8) {
    let mut builder = pretty_env_logger::formatted_timed_builder();

    let noisy_modules = &[
        "hyper",
        "mio",
        "tokio_core",
        "tokio_reactor",
        "tokio_threadpool",
        "fuse::request",
        "rusoto_core",
        "want",
        "tantivy",
    ];

    let log_level = match level {
        //0 => log::Level::Error,
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    if level > 1 && level < 4 {
        for module in noisy_modules {
            builder.filter_module(module, log::LevelFilter::Info);
        }
    }

    builder.filter_level(log_level);
    builder.format_timestamp_millis();
    //builder.format(|buf, record| writeln!(buf, "{}", record.args()));
    builder.init();
}
