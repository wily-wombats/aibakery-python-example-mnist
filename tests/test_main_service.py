from aib.main_service import predict


def test_predict():
    results = predict(
        {
            'x': [0., 2., 0., 8., 9., 0., 0., 0., 0., 13., 5., 14., 8.,
                  7., 0., 0., 0., 12., 5., 2., 0., 9., 0., 0., 0., 7.,
                  5., 0., 0., 3., 5., 0., 0., 3., 10., 0., 0., 2., 10.,
                  0., 0., 1., 13., 0., 0., 1., 12., 0., 0., 0., 5., 13.,
                  5., 9., 13., 0., 0., 0., 0., 9., 16., 16., 7., 0.]
        }
    )['results']

    assert len(results) == 1
    assert results[0]['key'] == 'predicted_number'
    assert results[0]['value'] == 0
    assert results[0]['meta']['predicted_probability'] > 0.5
