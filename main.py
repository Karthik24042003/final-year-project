import pytest
from main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if the home page returns status code 200."""
    response = client.get('/')
    assert response.status_code == 200

def test_predict(client):
    """Test if the prediction page returns status code 200."""
    response = client.post('/predict', data={'message': 'This is a fake news article'})
    assert response.status_code == 200
