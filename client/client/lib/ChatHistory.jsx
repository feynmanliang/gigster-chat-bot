import React, { PropTypes } from 'react';
import Message from './Message.jsx';
import Feature from './Feature.jsx';

const { array, arrayOf, shape, string } = PropTypes;

const propTypes = {
    blurbs: arrayOf(shape({
        user: string,
        msg: string,
        ts: string
    })),
    features: array
};

class ChatHistory extends React.Component {
    constructor() {
        super();
    }

    renderBlurb(blurb) {
        return <Message key={ `${Math.random().toString()} ${blurb.ts}` } message={ blurb }/>;
    }

    showMessages(blurbs) {
        return blurbs.map(this.renderBlurb);
    }

    showFeatures(features) {
        return features.map( (feat, i) => {
            return <Feature index={ i } key={`feature-${i}`} feature={feat}/>;
        });
    }

    render() {
        const { features, blurbs } = this.props;
        return (
            <div>
                <ul
                    id="all-messages"
                    className="all-messages">
                    { this.showMessages(blurbs) }
                </ul>
                <ul
                    id="all-features"
                    className="all-features">
                    { this.showFeatures(features) }
                </ul>
           </div>
        );
    }
}

ChatHistory.propTypes = propTypes;
export default ChatHistory;
