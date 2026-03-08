from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('execution', '0003_order_fill_position_event'),
    ]

    operations = [
        migrations.CreateModel(
            name='Position',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('symbol', models.CharField(db_index=True, max_length=50)),
                ('side', models.CharField(choices=[('long', 'Long'), ('short', 'Short'), ('none', 'None')], max_length=10)),
                ('qty', models.DecimalField(decimal_places=8, max_digits=18)),
                ('entry_price', models.DecimalField(decimal_places=8, max_digits=18)),
                ('mark_price', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('liquidation_price', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('unrealized_pnl', models.DecimalField(decimal_places=8, default=0, max_digits=18)),
                ('realized_pnl', models.DecimalField(decimal_places=8, default=0, max_digits=18)),
                ('leverage', models.DecimalField(decimal_places=2, default=1, max_digits=5)),
                ('margin_mode', models.CharField(default='cross', max_length=20)),
                ('option_type', models.CharField(blank=True, max_length=10)),
                ('strike', models.DecimalField(blank=True, decimal_places=2, max_digits=12, null=True)),
                ('expiry', models.DateField(blank=True, null=True)),
                ('synced_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('account', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='positions', to='execution.exchangeaccount')),
                ('intent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='positions', to='execution.executionintent')),
            ],
            options={
                'db_table': 'execution_position',
                'ordering': ['-synced_at'],
                'unique_together': {('account', 'symbol')},
            },
        ),
        migrations.CreateModel(
            name='ExecutionEvent',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('event_type', models.CharField(choices=[('intent_created', 'Intent Created'), ('risk_check_passed', 'Risk Check Passed'), ('risk_check_failed', 'Risk Check Failed'), ('order_submitted', 'Order Submitted'), ('order_filled', 'Order Filled'), ('order_cancelled', 'Order Cancelled'), ('order_rejected', 'Order Rejected'), ('order_error', 'Order Error'), ('position_opened', 'Position Opened'), ('position_closed', 'Position Closed'), ('stop_triggered', 'Stop Triggered'), ('reconciliation', 'Reconciliation'), ('manual_override', 'Manual Override')], max_length=30)),
                ('payload', models.JSONField(default=dict, help_text='Event-specific data')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('intent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='events', to='execution.executionintent')),
                ('order', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='events', to='execution.order')),
            ],
            options={
                'db_table': 'execution_event',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='executionevent',
            index=models.Index(fields=['event_type', 'created_at'], name='execution_e_event_t_789abc_idx'),
        ),
    ]
