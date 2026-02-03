from views_hydranet.utils.utils_internal_containers import ModelOutputs


def test_model_outputs_initialization():
    """Test that ModelOutputs initializes with default empty lists."""
    outputs = ModelOutputs()
    assert isinstance(outputs.y_score, list)
    assert len(outputs.y_score) == 0
    assert outputs.pg_id == []

def test_make_output_dict():
    """Test batch creation of output containers."""
    steps = 5
    out_dict = ModelOutputs.make_output_dict(steps=steps)

    assert len(out_dict) == steps
    assert "step01" in out_dict
    assert "step05" in out_dict
    assert isinstance(out_dict["step01"], ModelOutputs)

def test_output_dict_to_dataframe_explosion():
    """Test conversion of dictionary to DataFrame with list explosion."""
    dict_of_outputs = {
        "step01": ModelOutputs(
            y_score=[0.1, 0.2],
            y_true=[1.0, 0.0],
            pg_id=[100, 101]
        ),
        "step02": ModelOutputs(
            y_score=[0.3],
            y_true=[1.0],
            pg_id=[102]
        )
    }

    df = ModelOutputs.output_dict_to_dataframe(dict_of_outputs)

    # 2 rows for step01 + 1 row for step02 = 3 rows total
    assert len(df) == 3
    assert list(df["y_score"]) == [0.1, 0.2, 0.3]
    assert list(df["pg_id"]) == [100, 101, 102]
    # Note: apply(pd.Series.explode) often resets the index to integers, so we don't assert on labels.

def test_output_dict_to_dataframe_empty():
    """Test behavior with empty dictionary."""
    df = ModelOutputs.output_dict_to_dataframe({})
    assert df.empty
