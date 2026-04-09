import os
import sys
import pandas as pd
import requests

API_KEY = os.getenv('TICKETMASTER_API_KEY')
CITY = os.getenv('TM_CITY', 'Vancouver')
COUNTRY = os.getenv('TM_COUNTRY', 'CA')
START = os.getenv('TM_START', '2026-04-01T00:00:00Z')
END = os.getenv('TM_END', '2026-05-31T23:59:59Z')


def fetch_ticketmaster_events(api_key, start_dt, end_dt, city='Vancouver', country='CA'):
    url = 'https://app.ticketmaster.com/discovery/v2/events.json'
    params = {
        'apikey': api_key,
        'countryCode': country,
        'city': city,
        'startDateTime': start_dt,
        'endDateTime': end_dt,
        'size': 200,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    events = data.get('_embedded', {}).get('events', [])
    rows = []
    for ev in events:
        start = ev.get('dates', {}).get('start', {})
        date_str = start.get('localDate') or start.get('dateTime')
        if not date_str:
            continue
        event_date = pd.to_datetime(date_str)
        week_start = event_date - pd.to_timedelta(event_date.weekday(), unit='D')
        rows.append({
            'name': ev.get('name'),
            'event_date': pd.to_datetime(event_date).date().isoformat(),
            'week_start': pd.to_datetime(week_start).date().isoformat(),
            'id': ev.get('id'),
        })
    detail_df = pd.DataFrame(rows)
    weekly_df = pd.DataFrame(columns=['week_start', 'n_events'])
    if not detail_df.empty:
        weekly_df = detail_df.groupby('week_start').size().reset_index(name='n_events').sort_values('week_start')
    return data, detail_df, weekly_df


def main():
    if not API_KEY:
        print('Missing TICKETMASTER_API_KEY environment variable.')
        print('Run like: TICKETMASTER_API_KEY=your_key python output/check_events_api.py')
        sys.exit(1)

    try:
        data, detail_df, weekly_df = fetch_ticketmaster_events(API_KEY, START, END, CITY, COUNTRY)
    except Exception as e:
        print(f'API request failed: {e}')
        sys.exit(2)

    page = data.get('page', {})
    print(f'City: {CITY}, Country: {COUNTRY}')
    print(f'Window: {START} -> {END}')
    print(f"Ticketmaster page info: size={page.get('size')} totalElements={page.get('totalElements')} totalPages={page.get('totalPages')} number={page.get('number')}")
    print(f'Retrieved raw events in this response: {len(detail_df)}')
    print('')

    if detail_df.empty:
        print('No events retrieved in this window.')
        return

    print('Sample events:')
    print(detail_df[['event_date', 'week_start', 'name']].head(10).to_string(index=False))
    print('')
    print('Weekly counts:')
    print(weekly_df.to_string(index=False))


if __name__ == '__main__':
    main()
