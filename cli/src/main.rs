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
    let width = 10usize;
    let sigma = 1.0f32;

    // let npix = 2 * ((3.0 * sigma).ceil() as usize + 1);
    let npix = 4usize;
    let half = (npix / 2) as i32;
    // let mut kernel1D = Vec::<f32>::with_capacity(npix);

    let neghalfinvvar = -1.0 / (2.0 * sigma * sigma);

    let mut total = 0.0;
    let mut kernel1D = (0..npix)
        .map(|idx| {
            let dx = idx as f32 - 0.5 * (npix as f32 - 1.0);
            let k = (dx * dx * neghalfinvvar).exp();
            total += k;
            k
        })
        .collect::<Vec<_>>();
    // for (idx, k) in kernel1D.iter_mut().enumerate() {
    //     let dx = idx as f32 - 0.5 * (npix as f32 - 1.0);
    //     *k = (dx * dx * neghalfinvvar).exp();
    //     total += *k;
    // }
    let scale = 500.0 / total;
    for k in &mut kernel1D {
        *k *= scale;
    }

    log::info!("{:?}", kernel1D);

    log::info!("img_width:{} npix:{} half:{}", width, npix, half);

    for idx in 0..width as i32 {
        let start = std::cmp::max(0, idx - half);
        let end = std::cmp::min((width - 1) as i32, idx + half);
        log::debug!("start:{} end:{} run:{}", start, end, end - start);
        for sample in start..end {
            let base_idx = sample - idx + half;
            log::trace!("input[{}] * kernel[{}]", sample, base_idx);
        }
    }

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
