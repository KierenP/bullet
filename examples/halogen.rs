use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{Ranger, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{ViriBinpackLoader, viribinpack},
    },
};

fn main() {
    const NET_ID: &str = "bullet_r110-768x8hm-1536-dp-pw-16-da-32-1x8";

    // network hyperparams
    let ft_size = 1536;
    let l1_size = 16;
    let l2_size = 32;
    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ];
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let save_format = [
        SavedFormat::id("l0w")
            .transform(|store, weights| {
                let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .quantise::<i16>(255)
            .round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255).round(),
        SavedFormat::id("l2w").transpose().round(),
        SavedFormat::id("l2b").round(),
        SavedFormat::id("l3w").transpose().round(),
        SavedFormat::id("l3b").round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<8>)
        .optimiser(Ranger)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS * l1_size);
            let l2 = builder.new_affine("l2", l1_size * 2, NUM_OUTPUT_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);
            out = out.concat(out.abs_pow(2.0)).crelu();
            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);
            (out, loss)
        });

    // cap l1 weights to 1.98 after factoriser is applied
    let l0_params = RangerParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };

    // allow float weights to have a large range
    let float_params = RangerParams { max_weight: 128.0, min_weight: -128.0, ..Default::default() };

    trainer.optimiser.set_params_for_weight("l0w", l0_params);
    trainer.optimiser.set_params_for_weight("l0f", l0_params);
    trainer.optimiser.set_params_for_weight("l2w", float_params);
    trainer.optimiser.set_params_for_weight("l2b", float_params);
    trainer.optimiser.set_params_for_weight("l3w", float_params);
    trainer.optimiser.set_params_for_weight("l3b", float_params);

    // ============== STAGE 1: Main training on datagen3-22 ==============
    let stage1_superbatches = 1000;
    let stage1 = TrainingSchedule {
        net_id: format!("{NET_ID}-stage1"),
        eval_scale: 160.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: stage1_superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.7 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: stage1_superbatches },
        save_rate: 100,
    };

    let data_loader1 = ViriBinpackLoader::new(
        "..\\..\\chess\\data\\datagen3-22.viri",
        1024 * 32,
        4,
        viribinpack::ViriFilter::Builtin(viriformat::dataformat::Filter {
            min_ply: 0,
            min_pieces: 0,
            ..Default::default()
        }),
    );

    // ============== STAGE 2: Fine-tuning on datagen18-22 ==============
    let stage2_superbatches = 100;
    let stage2 = TrainingSchedule {
        net_id: format!("{NET_ID}-stage2"),
        eval_scale: 160.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: stage2_superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.0001, final_lr: 0.0, final_superbatch: stage2_superbatches },
        save_rate: 100,
    };

    let data_loader2 = ViriBinpackLoader::new(
        "..\\..\\chess\\data\\datagen18-22.viri",
        1024 * 32,
        4,
        viribinpack::ViriFilter::Builtin(viriformat::dataformat::Filter {
            min_ply: 0,
            min_pieces: 0,
            ..Default::default()
        }),
    );

    // ============== Run training pipeline ==============
    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    //trainer.run(&stage1, &settings, &data_loader1);
    trainer.load_from_checkpoint("checkpoints/bullet_r108-768x8hm-1536-dp-pw-16-da-32-1x8-1000");
    trainer.run(&stage2, &settings, &data_loader2);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 160.0 * eval);
    }

    trainer.save_quantised(&format!("nets/{NET_ID}-stage2-e{stage2_superbatches}.nn")).unwrap();
}
