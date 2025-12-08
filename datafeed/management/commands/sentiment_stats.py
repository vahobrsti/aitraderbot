from django.core.management.base import BaseCommand
from django.db.models import Min, Max, Avg, Count,StdDev
from datafeed.models import RawDailyData


class Command(BaseCommand):
    help = "Show min/max/avg/count for sentiment over ALL RawDailyData rows"

    def handle(self, *args, **options):
        field = "sentiment_weighted_total"  # change if needed

        qs = RawDailyData.objects.all()

        if not qs.exists():
            self.stdout.write(self.style.WARNING("No rows in RawDailyData"))
            return

        stats = qs.aggregate(
            min_value=Min(field),
            max_value=Max(field),
            avg_value=Avg(field),
            std_value=StdDev(field),
            row_count=Count("id"),
        )

        self.stdout.write(self.style.SUCCESS(f"Stats for {field} (ALL rows)"))
        self.stdout.write(f"Count: {stats['row_count']}")
        self.stdout.write(f"Min:   {stats['min_value']}")
        self.stdout.write(f"Max:   {stats['max_value']}")
        self.stdout.write(f"Avg:   {stats['avg_value']}")
        self.stdout.write(f"std:   {stats['std_value']}")
