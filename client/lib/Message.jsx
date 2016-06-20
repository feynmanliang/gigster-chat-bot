import React, { PropTypes } from 'react';

const { shape, string } = PropTypes;

const propTypes = {
    message: shape({
        user: string,
        msg: string,
        ts: string
    }),
    features: PropTypes.object
};

class Message extends React.Component {
    constructor() {
        super();
        this.colorMap = {
            me: "#D8D8D8",
            bot: "#52004f"
        };
    }

    getColorStyle(user) {
        return { borderLeftColor: this.colorMap[user] };
    }

    makeFeatures(user, features) {
      if (user === 'bot') {
        let features;
        try {
          features = this.props.features.data.map(x => x.data);
        } catch(err) {
          features = [0, 0];
        }
        features = JSON.stringify({
          marketplace: features[0],
          social: features[1]
        });

        return (
          <div className="features">
            {features}
          </div>
        );
      }
    }

    render() {
        const { message } = this.props;
        const { user, msg, ts } = message;

        return (
            <li
                style={this.getColorStyle(user)}
                className="user-message">
                <div className="message-field user-message">
                  {msg}
                  {this.makeFeatures(user)}
                </div>
            </li>
        );
    }
}

Message.propTypes = propTypes;
export default Message;
