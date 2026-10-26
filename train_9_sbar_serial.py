"""Nine-material serial energy -> damage experiment, using the d006 wide baseline."""
from train_9_sbar import main as run


DEFAULTS = dict(
    training_data='proportional', damage_input='free_energy',
    output_dir='outputs/9_sbar_serial_d006_plateau_wide',
    warmup_epochs=100, damage_epochs=1000, initial_damage=0.06,
    batch_size=256, learning_rate=0.001, warmup_learning_rate=None,
    lr_scheduler='plateau', lr_step_size=200, lr_gamma=0.5,
    lr_patience=40, lr_factor=0.5, lr_threshold=1e-4, min_lr=1e-5,
    seed=42, validation_fraction=0.2,
)


def main(arguments=None):
    run(arguments, defaults=DEFAULTS)


if __name__ == '__main__':
    main()
