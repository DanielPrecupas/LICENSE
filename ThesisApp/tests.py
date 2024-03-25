from django.test import TestCase
from django.urls import reverse
from .models import Flashcard
from .views import handle_csv

class FlashcardTests(TestCase):
    def setUp(self):
        self.flashcard = Flashcard.objects.create(
            question='What is the capital of France?',
            answer='Paris'
        )

    def test_flip_view(self):
        response = self.client.get(reverse('ThesisApp:flip', args=[self.flashcard.id]))
        print(response.content.decode('utf-8'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'What is the capital of France?')
        self.assertContains(response, 'Paris')


    def test_start_game_view(self):
        response = self.client.get(reverse('ThesisApp:start_game'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Start Studying')

    def test_game_view(self):
        response = self.client.get(reverse('ThesisApp:game'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Game')

    def test_handle_csv(self):
        csv_path = '/path/to/your/project/ThesisApp/Resources/data.csv'
        flashcard_list = handle_csv(csv_path)
        self.assertEqual(len(flashcard_list), 1)
        self.assertEqual(flashcard_list[0].question, 'Test Question')
        self.assertEqual(flashcard_list[0].answer, 'Test Answer')
