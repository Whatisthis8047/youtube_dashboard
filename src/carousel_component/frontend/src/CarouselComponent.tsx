import React from 'react';
import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from "streamlit-component-lib";

interface State {
  selectedTitle: string | null;
  selectedLink: string | null;
  selectedDescription: string | null;
  currentIndex: number;
}

interface CardProps {
  image: string;
  title: string;
  link: string;
  description?: string;
}

class CarouselComponent extends StreamlitComponentBase<State> {
  public state: State = { selectedTitle: null, selectedLink: null, selectedDescription: null, currentIndex: 0 };

  public render = (): React.ReactNode => {
    const { theme } = this.props;
    const cards: CardProps[] = this.props.args["cards"] || [];
    const layout = this.props.args["layout"] || 'default';

    const cardStyle: React.CSSProperties = {
      border: '1px solid #ddd',
      borderRadius: '0',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: layout === 'alternate' ? 'row' : 'column',
      height: layout === 'alternate' ? '150px' : 'auto',
    };

    const imageStyle: React.CSSProperties = {
      width: layout === 'alternate' ? '30%' : '100%',
      height: layout === 'alternate' ? '100%' : 'auto',
      objectFit: 'cover',
    };

    const contentStyle: React.CSSProperties = {
      padding: '10px',
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: layout === 'default' ? 'center' : 'flex-start',
    };

    const titleStyle: React.CSSProperties = {
      margin: '0 0 5px 0',
      fontSize: '16px',
      fontWeight: 'bold',
      textAlign: layout === 'default' ? 'center' : 'left',
    };

    const descriptionStyle: React.CSSProperties = {
      margin: '0',
      fontSize: '14px',
      color: '#666',
    };

    return (
      <div className="carousel-container">
        <button className="carousel-arrow carousel-arrow-left" onClick={this.handlePrev}>&lt;</button>
        <div className="carousel-content">
          {cards.slice(this.state.currentIndex, this.state.currentIndex + 3).map((card, index) => (
            <div key={index} className="carousel-card" style={cardStyle} onClick={() => this.handleCardClick(card, layout)}>
              <img src={card.image} alt={card.title} style={imageStyle} />
              <div style={contentStyle}>
                <h3 style={titleStyle}>{card.title}</h3>
                {layout === 'alternate' && card.description && (
                  <p style={descriptionStyle}>{card.description}</p>
                )}
              </div>
            </div>
          ))}
        </div>
        <button className="carousel-arrow carousel-arrow-right" onClick={this.handleNext}>&gt;</button>
      </div>
    );
  }

  private handleCardClick = (card: CardProps, layout: string): void => {
    if (layout === 'alternate') {
      this.setState(
        { selectedTitle: card.title, selectedDescription: card.description || null },
        () => Streamlit.setComponentValue({ title: card.title, description: card.description })
      );
    } else {
      this.setState(
        { selectedTitle: card.title, selectedLink: card.link },
        () => Streamlit.setComponentValue({ title: card.title, link: card.link })
      );
    }
  }

  private handleNext = (): void => {
    const cards: CardProps[] = this.props.args["cards"] || [];
    this.setState(prevState => ({
      currentIndex: (prevState.currentIndex + 1) % (cards.length - 2)
    }));
  }

  private handlePrev = (): void => {
    const cards: CardProps[] = this.props.args["cards"] || [];
    this.setState(prevState => ({
      currentIndex: (prevState.currentIndex - 1 + (cards.length - 2)) % (cards.length - 2)
    }));
  }
}

export default withStreamlitConnection(CarouselComponent);