from dvclive.lightning import DVCLiveLogger


class DVCLiveLoggerCloud(DVCLiveLogger):
    def __init__(
        self,
        run_name: Optional[str] = "dvclive_run",
        prefix="",
        log_model: Union[str, bool] = False,
        experiment=None,
        **kwargs,
    ):
        super().__init__(
            run_name=run_name,
            prefix=prefix,
            experiment=experiment,
            **kwargs,
        )
    def _upload_to_cloud(self, path: str) -> None:
        self.experiment.log_artifact(path)
    
    def _save_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # drop unused checkpoints
        if not self.experiment._resume and checkpoint_callback.dirpath:  # noqa: SLF001
            for p in Path(checkpoint_callback.dirpath).iterdir():
                if str(p) not in self._all_checkpoint_paths:
                    p.unlink(missing_ok=True)

        # save directory
        self.experiment.log_artifact(checkpoint_callback.dirpath)
