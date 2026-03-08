from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    initial = True
    dependencies = [
        ('signals', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ExchangeAccount',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='Friendly name for this account', max_length=100)),
                ('exchange', models.CharField(choices=[('bybit', 'Bybit'), ('deribit', 'Deribit')], max_length=20)),
                ('account_type', models.CharField(choices=[('unified', 'Unified Trading'), ('classic', 'Classic'), ('portfolio', 'Portfolio Margin')], default='unified', max_length=20)),
                ('api_key_env', models.CharField(help_text='Environment variable name for API key', max_length=100)),
                ('api_secret_env', models.CharField(help_text='Environment variable name for API secret', max_length=100)),
                ('is_testnet', models.BooleanField(default=True, help_text='Use testnet endpoints')),
                ('is_active', models.BooleanField(default=True)),
                ('max_position_usd', models.DecimalField(decimal_places=2, default=10000, help_text='Maximum position size in USD', max_digits=12)),
                ('max_daily_loss_usd', models.DecimalField(decimal_places=2, default=1000, help_text='Daily loss limit in USD', max_digits=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Exchange Account',
                'verbose_name_plural': 'Exchange Accounts',
                'db_table': 'exchange_account',
            },
        ),
    ]
