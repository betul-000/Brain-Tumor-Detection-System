# Generated by Django 4.1.2 on 2022-10-22 13:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sample', '0003_image_name'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Image',
        ),
    ]