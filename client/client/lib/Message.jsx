import React, { PropTypes } from 'react';

const { shape, string } = PropTypes;

const propTypes = {
    message: shape({
        user: string,
        msg: string,
        ts: string
    })
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

        return (
            <li
                style={this.getColorStyle(user)}
                className="user-message">
                <div className="message-field user-message">{msg}</div>
            </li>
        );
    }
}

Message.propTypes = propTypes;
export default Message;
