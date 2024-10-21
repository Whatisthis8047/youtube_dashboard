import os
import streamlit.components.v1 as components
import pandas as pd

_RELEASE = False

if not _RELEASE:
    _carousel_component = components.declare_component(
        "carousel_component",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _carousel_component = components.declare_component("carousel_component", path=build_dir)


def carousel_component(data, layout='default', key=None):
    try:
        if isinstance(data, pd.DataFrame):
            if layout == 'default':
                required_columns = ['title', 'link', 'thumbnail']
            else:  # 'alternate'
                required_columns = ['title', 'member', 'thumbnail', 'keyword']

            if not all(col in data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Input DataFrame is missing required columns: {', '.join(missing_cols)}")

            cards = data[required_columns].to_dict('records')

            if layout == 'default':
                cards = [{'title': card['title'], 'link': card['link'], 'image': card['thumbnail']} for card in cards]
            else:
                ## TO-DO 키워드 이쁘게 하기....

                cards = [{'title': card['member'], 'image': card['thumbnail'],
                          'description': ', '.join('#' + word.strip("'[]") for word in card['keyword'].split(', '))}
                         for card in cards]

        else:
            raise ValueError("Input must be a pandas DataFrame")
        component_value = _carousel_component(cards=cards, layout=layout, key=key, default=None)

        return component_value

    except Exception as e:
        print('error')
        return None