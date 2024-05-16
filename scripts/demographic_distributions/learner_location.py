import sqlite3
import pandas as pd
import plotly.express as px
from course_utilities import identify_course
import os
from dotenv import load_dotenv

load_dotenv()

BASE_FIGURE_URL = "figures/location/"

MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')


def fetch_country_gender_course_id() -> pd.DataFrame:
    conn = sqlite3.connect(MOOC_DB_LOCATION)
    cur = conn.cursor()

    cur.execute("""
                SELECT UP.country, UP.gender, E.course_id
                FROM user_profiles UP
                JOIN enrollments E ON UP.hash_id = E.hash_id
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Country", "Gender", "Course ID"])

    cur.close()
    conn.close()

    return df


def plot_location_distribution(country_df: pd.DataFrame) -> None:
    country_counts = country_df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    fig = px.choropleth(country_counts, locations='Country',
                        color='Count',
                        hover_name='Country',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        locationmode='ISO-3',
                        labels={'Count': 'Number of Learners'})

    fig.update_layout(
        showlegend=False,
        title_text='Learner location distribution',
        title_x=0.5,
        autosize=False,
        width=1300,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=0,
            t=50,
            pad=0
        )
    )
    fig.write_image(BASE_FIGURE_URL + "learner_location_distribution.png")


def plot_location_distribution_per_course(course_country_df: pd.DataFrame) -> None:
    for course in course_country_df['Course ID'].unique():
        course_df = course_country_df[course_country_df['Course ID'] == course]
        course_name = identify_course(course)
        country_counts = course_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        fig = px.choropleth(country_counts, locations='Country',
                            color='Count',
                            hover_name='Country',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            locationmode='ISO-3',
                            labels={'Count': 'Number of Learners'})

        fig.update_layout(
            showlegend=False,
            title_text=f'Learner location distribution for {course_name}',
            title_x=0.5,
            autosize=False,
            width=1300,
            height=600,
            margin=dict(
                l=50,
                r=50,
                b=0,
                t=50,
                pad=0
            )
        )
        fig.write_image(BASE_FIGURE_URL +
                        f"learner_location_distribution_{course_name}.png")


def plot_location_distribution_per_gender(location_df: pd.DataFrame, graph_name: str = None) -> None:
    men_df = location_df[location_df['Gender'] == 'm']
    women_df = location_df[location_df['Gender'] == 'f']

    if graph_name:
        print(graph_name)
        # Get top 5 countries by men + women count
        top_countries = location_df['Country'].value_counts().head(5).index
        for country in top_countries:
            country_df = location_df[location_df['Country'] == country]
            men_count = len(country_df[country_df['Gender'] == 'm'])
            women_count = len(country_df[country_df['Gender'] == 'f'])
            print(f"{country} total count {men_count + women_count}")
            print(f"{country} male count {men_count}")
            print(f"{country} female count {women_count}")

    men_counts = men_df['Country'].value_counts().reset_index()
    men_counts.columns = ['Country', 'Count']
    women_counts = women_df['Country'].value_counts().reset_index()
    women_counts.columns = ['Country', 'Count']

    fig_men = px.choropleth(men_counts, locations='Country',
                            color='Count',
                            hover_name='Country',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            locationmode='ISO-3',
                            labels={'Count': 'Number of Men'})

    fig_men.update_layout(
        showlegend=False,
        title_text=f'{
            graph_name} (Men)' if graph_name else 'Learner location distribution (Men)',
        title_x=0.5,
        autosize=False,
        width=1300,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=10,
            t=50,
            pad=0
        )
    )

    image_location = BASE_FIGURE_URL + \
        f"{graph_name} Men.png" if graph_name else BASE_FIGURE_URL + \
        "learner_location_distribution_men.png"
    fig_men.write_image(image_location)

    fig_women = px.choropleth(women_counts, locations='Country',
                              color='Count',
                              hover_name='Country',
                              color_continuous_scale=px.colors.sequential.Plasma,
                              locationmode='ISO-3',
                              labels={'Count': 'Number of Women'})
    fig_women.update_layout(
        showlegend=False,
        title_text=f'{
            graph_name} (Women)' if graph_name else 'Learner location distribution (Women)',
        title_x=0.5,
        autosize=False,
        width=1300,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=5,
            t=50,
            pad=0
        )
    )

    image_location = BASE_FIGURE_URL + \
        f"{graph_name} Women.png" if graph_name else "figures/learner_location_distribution_women.png"
    fig_women.write_image(image_location)


def plot_location_distribution_per_gender_per_course(location_gender_per_course_df: pd.DataFrame) -> None:
    location_gender_per_course_df['course_name'] = location_gender_per_course_df['Course ID'].apply(
        identify_course)

    for course_name in location_gender_per_course_df['course_name'].unique():
        course_df = location_gender_per_course_df[location_gender_per_course_df['course_name'] == course_name]
        graph_name = f"Learner location distribution for {course_name}"
        plot_location_distribution_per_gender(course_df, graph_name)


def plot_location_distribution_female_male_ratio(df: pd.DataFrame) -> None:
    men_df = df[df['Gender'] == 'm']
    women_df = df[df['Gender'] == 'f']

    men_counts = men_df['Country'].value_counts().reset_index()
    men_counts.columns = ['Country', 'Count']
    women_counts = women_df['Country'].value_counts().reset_index()
    women_counts.columns = ['Country', 'Count']

    merged = pd.merge(men_counts, women_counts, on='Country', how='outer')
    merged = merged.fillna(0)
    merged['Ratio'] = merged['Count_y'] / \
        (merged['Count_x'] + merged['Count_y'])
    fig_ratio = px.choropleth(merged, locations='Country',
                              color='Ratio',
                              hover_name='Country',
                              color_continuous_scale=px.colors.sequential.Plasma,
                              locationmode='ISO-3',
                              )
    fig_ratio.update_layout(
        showlegend=False,
        title='Female-male ratio',
        title_x=0.5,
        autosize=False,
        width=1300,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=5,
            t=50,
            pad=0
        )
    )
    fig_ratio.write_image(
        BASE_FIGURE_URL + "learner_location_female_male_ratio.png")


def plot_location_distribution_female_male_ratio_per_course(location_gender_per_course_df: pd.DataFrame) -> None:
    for course in location_gender_per_course_df['Course ID'].unique():
        course_df = location_gender_per_course_df[location_gender_per_course_df['Course ID'] == course]
        course_name = identify_course(course)
        men_df = course_df[course_df['Gender'] == 'm']
        women_df = course_df[course_df['Gender'] == 'f']

        men_counts = men_df['Country'].value_counts().reset_index()
        men_counts.columns = ['Country', 'Count']
        women_counts = women_df['Country'].value_counts().reset_index()
        women_counts.columns = ['Country', 'Count']

        merged = pd.merge(men_counts, women_counts, on='Country', how='outer')
        merged = merged.fillna(0)
        merged['Ratio'] = merged['Count_y'] / \
            (merged['Count_x'] + merged['Count_y'])
        fig_ratio = px.choropleth(merged, locations='Country',
                                  color='Ratio',
                                  hover_name='Country',
                                  color_continuous_scale=px.colors.sequential.Plasma,
                                  locationmode='ISO-3',
                                  )
        fig_ratio.update_layout(
            showlegend=False,
            title=f'Female-male ratio for {course_name}',
            title_x=0.5,
            autosize=False,
            width=1300,
            height=600,
            margin=dict(
                l=50,
                r=50,
                b=5,
                t=50,
                pad=0
            )
        )
        fig_ratio.write_image(
            BASE_FIGURE_URL + f"learner_location_female_male_ratio_{course_name}.png")


def main() -> None:
    df = fetch_country_gender_course_id()
    # Country codes tsv created from https://www.nationsonline.org/oneworld/country_code_list.htm.
    country_codes = pd.read_csv(
        './country_codes.tsv', sep='\t', usecols=[0, 1, 2])
    country_codes.columns = ['Full name',
                             'Two-letter code', 'Three-letter code']
    df = df.merge(country_codes, left_on='Country',
                  right_on='Two-letter code', how='left')
    df = df.drop(columns=['Two-letter code', 'Country'])
    df = df.rename(columns={'Three-letter code': 'Country'})

    plot_location_distribution(df)
    plot_location_distribution_per_course(df)
    plot_location_distribution_per_gender(df)
    plot_location_distribution_per_gender_per_course(df)
    plot_location_distribution_female_male_ratio(df)
    plot_location_distribution_female_male_ratio_per_course(df)


if __name__ == '__main__':
    main()
