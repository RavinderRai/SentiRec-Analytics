from django.db import models

class Headphone(models.Model):
    name = models.CharField(max_length=100)
    image_url = models.URLField()

class Review(models.Model):
    headphone = models.ForeignKey(Headphone, on_delete=models.CASCADE)
    summary = models.TextField()