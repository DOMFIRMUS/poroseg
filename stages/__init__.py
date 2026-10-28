"""Analysis stages. Each module exposes ``run(ctx: poroseg.context.Context) -> None``.

Order (see poroseg.run.STAGES):
segmentation -> descriptive -> spatial -> sensitivity -> perturbation ->
morphology -> synthetic -> inference -> legacy -> report
"""
