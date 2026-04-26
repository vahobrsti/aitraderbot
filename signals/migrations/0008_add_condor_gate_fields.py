"""Add iron condor gate fields to DailySignal."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("signals", "0007_add_numeric_execution_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="dailysignal",
            name="condor_score",
            field=models.FloatField(default=0.0, help_text="Range score (0-100) for iron condor eligibility"),
        ),
        migrations.AddField(
            model_name="dailysignal",
            name="condor_eligible",
            field=models.BooleanField(default=False, help_text="Whether condor gate passed (score >= threshold, no vetoes)"),
        ),
        migrations.AddField(
            model_name="dailysignal",
            name="condor_veto_reasons",
            field=models.JSONField(blank=True, default=list, help_text="Hard veto reasons blocking condor entry"),
        ),
        migrations.AddField(
            model_name="dailysignal",
            name="condor_score_components",
            field=models.JSONField(blank=True, default=dict, help_text="Breakdown of range score components"),
        ),
    ]
