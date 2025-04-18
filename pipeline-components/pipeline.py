import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output

@dsl.container_component
def data_proc_op(
    # Define output as a parameter
    train_df: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_data_proc:latest',
        command=['python', '-u', 'data_proc.py'],
        args=[
            '--train_dir', '/app/data',
            '--output_name', train_df.path  # Use the output path directly
        ]
    )


@dsl.container_component
def data_proc_alt_op(
    train_df_alt: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_data_proc_alt:latest',
        args=[
            '--train_dir', '/app/data',
            '--output_name', train_df_alt.path
        ]
    )


@dsl.container_component
def feature_eng_op(
    # Define input and outputs as parameters
    train_df: Input[Dataset],
    train_df_engd: Output[Dataset],
    base_engd: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_feature_eng:latest',
        command=['python', '-u', 'feature_eng.py'],
        args=[
            '--data_path', train_df.path,
            # '--alt_data_path', '/app/data/train_df_alt.parquet',
            '--engd_data_path', train_df_engd.path,
            '--base_engd_path', base_engd.path,
        ]
    )


@dsl.container_component
def hyperopt_op(
    train_df_engd: Input[Dataset],
    base_engd: Input[Dataset]
):
    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_train:latest',
        command=['python', '-u', 'train.py'],
        args=[
            '--data_path', train_df_engd.path,
            '--base_data_path', base_engd.path,
        ]
    )


@dsl.container_component
def train_eval_op(
    train_df_engd: Input[Dataset],
    base_engd: Input[Dataset],
    trained_model: Output[Dataset]  # Added output for the model
):
    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_train:latest',
        command=['python', '-u', 'train.py'],
        args=[
            '--data_path', train_df_engd.path,
            '--base_data_path', base_engd.path,
            '--model_save_path', trained_model.path,
        ]
    )


@dsl.pipeline(
    name='Kaggle Credit Score',
    description='Kubeflow pipeline of kaggle Credit Score competition '
)
def kaggle_credit_pipeline():
    _data_proc_op = data_proc_op()

    # _data_proc_alt_op = data_proc_alt_op()

    _feature_eng_op = feature_eng_op(
        train_df=_data_proc_op.outputs["train_df"]
    ).after(_data_proc_op)

    _hyperopt_op = hyperopt_op(
        train_df_engd=_feature_eng_op.outputs["train_df_engd"],
        base_engd=_feature_eng_op.outputs["base_engd"]
    ).after(_feature_eng_op)

    _train_eval_op = train_eval_op(
        train_df_engd=_feature_eng_op.outputs["train_df_engd"],
        base_engd=_feature_eng_op.outputs["base_engd"]
    ).after(_hyperopt_op)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(kaggle_credit_pipeline, __file__[:-3] + '.yaml')

"""
import kfp
from kfp import dsl

@dsl.container_component
def data_proc_op():

    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_data_proc:latest',
        command=['python', '-u', 'data_proc.py'],
        args=[
            '--train_dir', '/app/data',
            '--output_name', 'train_df.parquet'
        ],
        file_outputs={'train_df': '/app/data/train_df.parquet'}
    )


@dsl.container_component
def data_proc_alt_op(train_df_alt: dsl.OutputPath(str)):

    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_data_proc_alt:latest',
        args=[
            '--train_dir', '/app/data',
            '--output_name', 'train_df_alt.parquet'
        ]
    )


@dsl.container_component
def feature_eng_op(train_df):

    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_feature_eng:latest',
        command=['python', '-u', 'feature_eng.py'],
        args=[
            '--data_path', dsl.InputArtifactArgument(train_df),
            # '--alt_data_path', '/app/data/train_df_alt.parquet',
            '--engd_data_path', '/app/data/train_df_engd.parquet',
            '--base_engd_path', '/app/data/base_engd.parquet',
        ],
        file_outputs={
            'train_df_engd': '/app/data/train_df_engd.parquet',
            'base_engd': '/app/data/base_engd.parquet'
        }
    )

@dsl.container_component
def hyperopt_op(train_df_engd, base_engd):

    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_train:latest',
        command=['python', '-u', 'train.py'],
        args=[
            '--data_path', dsl.InputArtifactArgument(train_df_engd),
            '--base_data_path', dsl.InputArtifactArgument(base_engd),
        ]
    )

@dsl.container_component
def train_eval_op(train_df_engd, base_engd):

    return dsl.ContainerSpec(
        image='aldarionthemariner/kaggle_train:latest',
        command=['python', '-u', 'train.py'],
        args=[
            '--data_path', dsl.InputArtifactArgument(train_df_engd),
            '--base_data_path', dsl.InputArtifactArgument(base_engd),
            '--model_save_path', '/app/lgb',
        ],
    )


@dsl.pipeline(
    name='Kaggle Credit Score',
    description='Kubeflow pipeline of kaggle Credit Score competition '
)
def kaggle_credit_pipeline():
    _data_proc_op = data_proc_op()

    # _data_proc_alt_op = data_proc_alt_op().after(_data_proc_op)

    _feature_eng_op = feature_eng_op(
        train_df=_data_proc_op.outputs['train_df']
    ).after(_data_proc_op)

    _hyperopt_op = hyperopt_op(
        train_df_engd=_feature_eng_op.outputs['train_df_engd'],
        base_engd=_feature_eng_op.outputs['base_engd']
    ).after(_feature_eng_op)

    _train_eval_op = train_eval_op(
        train_df_engd=_feature_eng_op.outputs['train_df_engd'],
        base_engd=_feature_eng_op.outputs['base_engd']
    ).after(_hyperopt_op)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(kaggle_credit_pipeline, __file__[:-3] + '.yaml')
"""