from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('execution', '0002_executionintent_order_fill_position_executionevent'),
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('exchange_order_id', models.CharField(blank=True, db_index=True, max_length=100)),
                ('client_order_id', models.CharField(max_length=100, unique=True)),
                ('symbol', models.CharField(max_length=50)),
                ('side', models.CharField(choices=[('buy', 'Buy'), ('sell', 'Sell')], max_length=10)),
                ('order_type', models.CharField(choices=[('market', 'Market'), ('limit', 'Limit'), ('stop_market', 'Stop Market'), ('stop_limit', 'Stop Limit'), ('take_profit', 'Take Profit')], max_length=20)),
                ('qty', models.DecimalField(decimal_places=8, max_digits=18)),
                ('price', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('trigger_price', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('reduce_only', models.BooleanField(default=False)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('submitted', 'Submitted'), ('open', 'Open'), ('partial', 'Partially Filled'), ('filled', 'Filled'), ('cancelled', 'Cancelled'), ('rejected', 'Rejected'), ('expired', 'Expired')], default='pending', max_length=20)),
                ('filled_qty', models.DecimalField(decimal_places=8, default=0, max_digits=18)),
                ('avg_fill_price', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('error_code', models.CharField(blank=True, max_length=50)),
                ('error_message', models.TextField(blank=True)),
                ('retry_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('submitted_at', models.DateTimeField(blank=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('intent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='orders', to='execution.executionintent')),
            ],
            options={
                'db_table': 'execution_order',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='Fill',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('exchange_fill_id', models.CharField(max_length=100, unique=True)),
                ('qty', models.DecimalField(decimal_places=8, max_digits=18)),
                ('price', models.DecimalField(decimal_places=8, max_digits=18)),
                ('fee', models.DecimalField(decimal_places=8, default=0, max_digits=18)),
                ('fee_currency', models.CharField(default='USDT', max_length=10)),
                ('filled_at', models.DateTimeField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('order', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='fills', to='execution.order')),
            ],
            options={
                'db_table': 'execution_fill',
                'ordering': ['-filled_at'],
            },
        ),
    ]
