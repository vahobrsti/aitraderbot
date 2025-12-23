# api/management/commands/create_api_token.py
"""
Create or retrieve an API token for a user.
Use this to generate tokens for the Telegram bot.
"""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from rest_framework.authtoken.models import Token


class Command(BaseCommand):
    help = "Create or retrieve an API token for a user (creates user if needed)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--username",
            type=str,
            default="telegram_bot",
            help="Username for the token holder (default: telegram_bot)",
        )
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing token and create a new one",
        )

    def handle(self, *args, **options):
        username = options["username"]
        User = get_user_model()
        
        # Get or create user
        user, created = User.objects.get_or_create(
            username=username,
            defaults={"is_active": True}
        )
        
        if created:
            self.stdout.write(f"Created user: {username}")
        
        # Handle reset
        if options["reset"]:
            Token.objects.filter(user=user).delete()
            self.stdout.write("Deleted existing token")
        
        # Get or create token
        token, created = Token.objects.get_or_create(user=user)
        
        if created:
            self.stdout.write(self.style.SUCCESS(f"Created new token"))
        else:
            self.stdout.write(f"Retrieved existing token")
        
        self.stdout.write("")
        self.stdout.write("=" * 50)
        self.stdout.write(self.style.SUCCESS(f"API Token: {token.key}"))
        self.stdout.write("=" * 50)
        self.stdout.write("")
        self.stdout.write("Use in requests with header:")
        self.stdout.write(f"  Authorization: Token {token.key}")
        self.stdout.write("")
