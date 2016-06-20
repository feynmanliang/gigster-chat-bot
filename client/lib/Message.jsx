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

    render() {
        const { message } = this.props;
        const { user, msg, ts } = message;

        let features;
        try {
          features = this.props.features.data.map(x => x.data);
        } catch(err) {
          features = [0, 0];
        }
        features = JSON.stringify(features);

        return (
            <li
                style={this.getColorStyle(user)}
                className="user-message">
                <div className="message-field user-message">
                  {msg}
                  {features}
                </div>
            </li>
        );
    }
}

Message.propTypes = propTypes;
export default Message;
